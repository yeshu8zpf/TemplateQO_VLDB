import os

from matplotlib import pyplot as plt
from model_rank import TemplateAwarePlanSelector2
from dataset_rank import TemplateDataset, collate_fn
from preprocess_rank import load_data, prepare_data_multi_template, get_all_template_id, get_max_num_columns, get_filter_col_types, \
save_checkpoint, log_wandb, prepare_weights
from train_rank import *
import json, pickle, logging, yaml
import random, numpy as np, torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
import wandb
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()  
parser.add_argument('--database', type=str, default='imdb')
args = parser.parse_args()

# Load configuration from a YAML file
database = args.database
with open(f'configs/{database}.yaml', 'r') as f:
    config = EasyDict(yaml.safe_load(f))

# Set up logging and WandB if enabled in the config
if config.log.wandb:
    os.makedirs(f'logs/{config.log.database}/wandb', exist_ok=True)
    wandb.init(project='templateQO', name=str(config.log.run), dir=f'logs/{config.log.database}/wandb', config=config)
    table = wandb.Table(columns=["epoch", "valid_latency", "test_latency"])

else:
    os.makedirs(f'logs/{config.log.database}/tensorboard', exist_ok=True)
    writer = SummaryWriter(f'{config.log.log_path}/tensorboard/{config.log.run}')

# Set up basic logging configuration
logging.basicConfig(
    filename=os.path.join(config.log.log_path, config.log.database),  # Log file name
    level=logging.INFO,  # Set log level
    format='%(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)
logger = logging.getLogger('mylogger')

# Set the random seed for reproducibility
def set_seed(seed=0):
    random.seed(42)

    # Set numpy seed
    np.random.seed(42)

    # Set torch seed
    torch.manual_seed(42)

    # If using GPU, set GPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def main():
    set_seed(config.seed)  # Set the random seed for reproducibility
    prepare = config.prepare.prepare
    
    # Get configuration values
    max_num_plans = config.model.max_num_plans
    col_encoding_leng = config.model.col_encoding_leng

    train_save_dir = config.prepare.train_save_dir
    test_save_dir = config.prepare.test_save_dir
    train_data_dir = config.prepare.train_data_dir
    test_data_dir = config.prepare.test_data_dir
    valid_data_dir = config.prepare.valid_data_dir
    valid_save_dir = config.prepare.valid_save_dir
    used_templates = config.prepare.used_templates
    print('used_templates:', used_templates)

    # Load filter columns and range dictionary
    with open(config.prepare.filter_columns_path, 'r') as f:
        filter_columns = json.load(f)
    with open(config.prepare.range_dict_path, 'r') as f:
        range_dict = json.load(f)
    
    # Get maximum number of columns and column types
    max_num_column = get_max_num_columns(filter_columns)
    filter_col_types = get_filter_col_types(range_dict, filter_columns)
    
    # Initialize model
    model = TemplateAwarePlanSelector2(
        hidden_dim=config.model.hidden_dim,
        n_tables=len(filter_columns),
        template_ids=get_all_template_id(train_data_dir, config.prepare.used_templates),
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        max_num_column=max_num_column,
        num_classes=max_num_plans,
        col_encoding_leng=col_encoding_leng,
        filter_col_types=filter_col_types,
        table_index_encoding_len=config.model.table_index_encoding_len
    )

     # Move model to device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    num_epochs = config.train.num_epochs
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=float(config.train.lr))
    scheduler = None  # CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # If fine-tuning, load checkpoint
    if config.finetune.finetune_path:
        print(f'Load checkpoint from {config.finetune.finetune_path}')
        checkpoint = torch.load(config.finetune.finetune_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if config.finetune.fix_parameter:
            model.fix_parameters(checkpoint['other_attr_dict'], config.finetune.new_templates)
        optimizer.load_state_dict(checkpoint['opt_state_dict'])

    # Prepare data for training, validation, and testing
    logger.info('preparing data...')
    print('preparing data...')
    if prepare:
        # Prepare data for multiple templates
        train_sql_tables, train_table_column_encodings, train_labels, train_qids, train_template_ids, train_masks, train_weights = \
            prepare_data_multi_template(train_data_dir, train_save_dir, max_num_plans, range_dict, filter_columns, encoding_leng=col_encoding_leng,
                                     used_templates=used_templates, weighting_type=config.prepare.weighting_type, smooth_factor=config.prepare.smooth_factor)
        test_sql_tables, test_table_column_encodings, test_labels, test_qids, test_template_ids, test_masks, test_weights = \
            prepare_data_multi_template(test_data_dir, test_save_dir, max_num_plans, range_dict, filter_columns, encoding_leng=col_encoding_leng,
                                     used_templates=used_templates, weighting_type=config.prepare.weighting_type, smooth_factor=config.prepare.smooth_factor)
        if config.prepare.valid:
            valid_sql_tables, valid_table_column_encodings, valid_labels, valid_qids, valid_template_ids, valid_masks, valid_weights = \
            prepare_data_multi_template(valid_data_dir, valid_save_dir, max_num_plans, range_dict, filter_columns, encoding_leng=col_encoding_leng,
                                     used_templates=used_templates, weighting_type=config.prepare.weighting_type, smooth_factor=config.prepare.smooth_factor)
    else:
        # Load data from preprocessed files
        train_sql_tables, train_table_column_encodings, train_labels, train_qids, train_template_ids, train_masks, train_weights = load_data(train_save_dir)
        test_sql_tables, test_table_column_encodings, test_labels, test_qids, test_template_ids, test_masks, test_weights = load_data(test_save_dir)
        if config.prepare.prepare_weights:
            train_weights = prepare_weights(train_data_dir, used_templates, config.prepare.weighting_type, config.prepare.smooth_factor)
            test_weights = prepare_weights(test_data_dir, used_templates, config.prepare.weighting_type, config.prepare.smooth_factor)
            
        if config.prepare.valid:
            valid_sql_tables, valid_table_column_encodings, valid_labels, valid_qids, valid_template_ids, valid_masks, valid_weights = load_data(valid_save_dir)
    print('finish load raw data...')

    # Create dataset and dataloaders
    train_dataset = TemplateDataset(train_sql_tables, train_table_column_encodings, train_template_ids, train_labels, train_qids, train_masks, train_weights)
    test_dataset = TemplateDataset(test_sql_tables, test_table_column_encodings, test_template_ids, test_labels, test_qids, test_masks, test_weights)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
    if config.prepare.valid:
        valid_dataset = TemplateDataset(valid_sql_tables, valid_table_column_encodings, valid_template_ids, valid_labels, valid_qids, valid_masks, valid_weights)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        valid_dataloader = None

    # Start training
    logger.info('start training')
    print('start training...')

    first_template_latencys = []
    
    if len(used_templates) > 0:
        first_template_id = int(config.prepare.used_templates[0].lstrip('template'))
    else:
        for _, dirnames, _ in os.walk(train_data_dir):
            first_template_id = int(dirnames[0].lstrip('template'))
            break

    latency_sum_list = []
    valid_latency_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        if (epoch+1) % 1 == 0:
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}", extra={'end':''})
        train_loss = train_model(model, train_dataloader, criterion, optimizer, device, num_epochs=1, cur_epoch=epoch, scheduler=scheduler)

        if epoch % 1 ==0:
            test_loss, pairwise_accuracy, latency_sum, template_accuracies, best_acc, template_latency, best_true_latency, valid_latency \
                = test_model(model, test_dataloader, criterion, device, valid_dataloader)
            latency_sum_list.append(latency_sum)
            valid_latency_list.append(valid_latency)
            test_acc_list.append(best_acc)
            table.add_data(epoch, valid_latency, latency_sum)
        if (epoch + 1) % config.train.ckpt_epoch == 0:
            save_checkpoint(f'{config.log.log_path}/{config.log.database}/model', model, optimizer, epoch, first_template_id)
        if config.log.wandb:
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss, 'pairwise_accuracy': pairwise_accuracy,
                    'latency_sum': latency_sum, 'test_accuracy': best_acc, 'template_latency': template_latency,
                    'valid_latency': valid_latency, 'best_true_latency': best_true_latency})
        if config.log.tensorboard:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('pairwise_accuracy', pairwise_accuracy, epoch)
            writer.add_scalar('latency_sum', latency_sum, epoch)
            writer.add_scalar('test_accuracy', best_acc, epoch)
            writer.add_scalar('template_latency', template_latency, epoch)
            writer.add_scalar('valid_latency', valid_latency, epoch)
            writer.add_scalar('best_true_latency', best_true_latency, epoch)
    
    if config.log.wandb:
        wandb.finish()
    if config.log.tensorboard:
        writer.close()


if __name__ == "__main__":
    main()
