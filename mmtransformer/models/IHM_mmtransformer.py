# Main file for running the code
import os
import sys

from matplotlib.pyplot import step
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mimic3-benchmarks')))
from mimic3models import common_utils
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.in_hospital_mortality import utils as ihm_utils
'''
In intensive care units, where patients come in with a wide range of health conditions, 
triaging relies heavily on clinical judgment. ICU staff run numerous physiological tests, 
such as bloodwork and checking vital signs, 
to determine if patients are at immediate risk of dying if not treated aggressively.
'''

import utils
import Models
import pickle
import numpy as np
import random
import os

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score

from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

import warnings
import time


def init_all(model, init_func, *params, **kwargs):
    # Initialize all weights using init_func
    for p in model.parameters():
        init_func(p, *params, **kwargs)

 
def Evaluate(Labels, Preds, PredScores):
    # Get the evaluation metrics like AUC, percision and etc.
    precision, recall, fscore, support = precision_recall_fscore_support(Labels, Preds, average='binary')
    _, _, fscore_weighted, _ = precision_recall_fscore_support(Labels, Preds, average='weighted')
    accuracy = accuracy_score(Labels, Preds)
    confmat = confusion_matrix(Labels, Preds)
    sensitivity = confmat[0,0]/(confmat[0,0]+confmat[0,1])
    specificity = confmat[1,1]/(confmat[1,0]+confmat[1,1])
    roc_macro, roc_micro, roc_weighted = roc_auc_score(Labels, PredScores, average='macro'), roc_auc_score(Labels, PredScores, average='micro'), roc_auc_score(Labels, PredScores, average='weighted')
    prf_test = {'precision': precision, 'recall': recall, 'fscore': fscore, 'fscore_weighted': fscore_weighted, 'accuracy': accuracy, 'confusionMatrix': confmat, 'sensitivity': sensitivity, 'specificity': specificity, 'roc_macro': roc_macro, 'roc_micro': roc_micro, 'roc_weighted': roc_weighted}
    return prf_test


def Evaluate_Model(model, batch, names):
    # Test the model
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        FirstTime = False
        ema_loss = None
        eval_obj = utils.Eval_Metrics()
        
        PredScores = None
        for idx, sample in enumerate(batch):
            X = torch.tensor(sample[0], dtype=torch.float).to(device)
            y = torch.tensor(sample[1], dtype=torch.float).to(device)

            text = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[2]]
            attn = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[3]]
            times = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[4]]

            if sample[2].shape[0] == 0:
                continue
            # Run the model
            Logits, Probs = model(X, text, attn, times)

            Lambd = torch.tensor(0.01).to(device)
            l2_reg = model.get_l2()
            loss = model.criterion(Logits, y)
            loss += Lambd * l2_reg
            epoch_loss += loss.item() * y.size(0)
            predicted = Probs.data > 0.5
            if not FirstTime:
                PredScores = Probs
                TrueLabels = y
                PredLabels = predicted
                FirstTime = True
            else:
                PredScores = torch.cat([PredScores, Probs])
                TrueLabels = torch.cat([TrueLabels, y])
                PredLabels = torch.cat([PredLabels, predicted])
            eval_obj.add(Probs.detach().cpu(), y.detach().cpu())

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                alpha = 0.01
                ema_loss = alpha * loss.item() + (1 - alpha) * ema_loss

            # writer.add_scalar(f"{MODEL_NAME_OUT}/Test/Loss/Details", loss.item(), idx)
            writer.add_scalars(f"{MODEL_NAME_OUT}/Test/Loss/Details", {"Batch-wise": loss.item()}, idx)
            writer.add_scalars(f"{MODEL_NAME_OUT}/Test/Loss/Details", {"EMA": ema_loss}, idx)
            
        prf_test = Evaluate(TrueLabels.detach().cpu(), PredLabels.detach().cpu(), PredScores.detach().cpu())
        prf_test['epoch_loss'] = epoch_loss / TrueLabels.shape[0]
        prf_test['aucpr'] = eval_obj.get_aucpr()
        
        # prf_test['acc'] = eval_obj.get_acc()
        # prf_test['recall'] = eval_obj.get_recall()
        # prf_test['precision'] = eval_obj.get_precision()
        # # prf_test['f1'] = eval_obj.get_f1()
        # prf_test['sensitivity'] = eval_obj.get_sensitivity()
        # prf_test['specificity'] = eval_obj.get_specificity()

    return prf_test, PredScores


def train_step(model, batches, epoch, num_epochs, display_step, optimizer):
    """
    Train the given model for 1 epoch
    """
    epoch_loss = 0
    total_step = len(batches)
    model.train()
    FirstTime = False
    ema_loss = None
    # Forward pass
    with torch.autograd.set_detect_anomaly(True):  # Error catcher
        for step, batch in enumerate(batches):
            torch.cuda.empty_cache()

            x_slice, y_slice, text_slice, attn_slice, time_slice = batch
            X = torch.tensor(x_slice, dtype=torch.float).to(device)
            y = torch.tensor(y_slice, dtype=torch.float).to(device)
            text = [torch.tensor(x, dtype=torch.long).to(device) for x in text_slice]
            attn = [torch.tensor(x, dtype=torch.long).to(device) for x in attn_slice]
            times = [torch.tensor(x, dtype=torch.float).to(device) for x in time_slice]

            Logits, Probs = model(X, text, attn, times)
            
            Lambd = torch.tensor(0.01).to(device)
            l2_reg = model.get_l2()

            loss = model.criterion(Logits, y)
            loss += Lambd * l2_reg
            with torch.no_grad():
                predicted = Probs.data > 0.5
                if not FirstTime:
                    PredScores = Probs
                    TrueLabels = y
                    PredLabels = predicted
                    FirstTime = True
                else:
                    PredScores = torch.cat([PredScores, Probs])
                    TrueLabels = torch.cat([TrueLabels, y])
                    PredLabels = torch.cat([PredLabels, predicted])
                epoch_loss += loss.item() * y.size(0)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # writer.add_scalar(f"Train/Loss[{epoch+1}/{num_epochs}]", loss.item(), global_step)
            # writer.add_scalar(f"Train/RegLoss[{epoch+1}/{num_epochs}]", (Lambd * l2_reg).item(), global_step)
            
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                alpha = 0.01
                ema_loss = alpha * loss.item() + (1 - alpha) * ema_loss

            writer.add_scalars(f"{MODEL_NAME_OUT}/Train/Loss/Details", { f"Epoch{epoch}": loss.item() }, step)
            writer.add_scalars(f"{MODEL_NAME_OUT}/Train/RegLoss/Details", { f"Epoch{epoch}": (Lambd * l2_reg).item() }, step)
            writer.add_scalars(f"{MODEL_NAME_OUT}/Train/Loss/Details/EMA", {"EMA": ema_loss}, step)

            optimizer.step()

            if (step + 1) % display_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, reg_loss: {:.4f}'.format(epoch + 1, num_epochs, step + 1, total_step, loss.item(), Lambd * l2_reg))

    with torch.no_grad():
        prf_train = Evaluate(TrueLabels.detach().cpu(), PredLabels.detach().cpu(), PredScores.detach().cpu())
        prf_train['epoch_loss'] = epoch_loss / TrueLabels.shape[0]
    return prf_train


def tokenizeGetIDs(tokenizer, text_data, max_len):
    # Tokenize the texts using tokenizer also pad to max_len and add <cls> token to first and <sep> token to end
    input_ids = []
    attention_masks = []
    for texts in text_data:
        # text_data (n_patient, time-stamp); texts: (time-stamps, ), all times along with all notes
        Textarr = [] 
        Attnarr = []
        for text in texts:
            # text: all notes for single time stamp
            # Textarr (time-stamps, max_len)
            encoded_sent = tokenizer.encode_plus(
                    text=text,                      # Preprocess sentence
                    add_special_tokens=True,        # Add [CLS] and [SEP]
                    max_length=max_len,             # Max length to truncate/pad
                    padding='max_length',           # Pad sentence to max length
                    #return_tensors='pt',           # Return PyTorch tensor
                    return_attention_mask=True,     # Return attention mask
                    truncation=True
                    )
            Textarr.append(encoded_sent.get('input_ids'))
            Attnarr.append(encoded_sent.get('attention_mask'))

    
        # Add the outputs to the lists
        # input_ids.append(encoded_sent.get('input_ids'))
        # attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids.append(Textarr)
        attention_masks.append(Attnarr)

    return input_ids, attention_masks


def concat_text_timeseries(data_reader, data_raw):
    train_text, train_times, start_time = data_reader.read_all_text_append_json(
        data_raw['names'], 
        48, 
        NumOfNotes = ARGS['NumOfNotes'])
    
    # Merge the text data with time-series data        
    data = utils.merge_text_raw(train_text, data_raw, train_times, start_time)
    return data


def get_time_to_end_diffs(times, starttimes):
    timetoends = []
    for times, st in zip(times, starttimes):
        difftimes = []
        et = np.datetime64(st) + np.timedelta64(49, 'h')

        for t in times:
            time = np.datetime64(t)
            dt = utils.diff_float(time, et)
            assert dt >= 0 #delta t should be positive
            difftimes.append(dt)
        timetoends.append(difftimes)
    return timetoends


def Read_Aggregate_data(mode, AggeragetNotesStrategies, discretizer=None, normalizer = None):
    # mode is between train, test, val
    # Build readers, discretizers, normalizers
    
    pretrained_data_path = os.path.join(CFG.pretrained_data_dir, mode + '_data_' +  AggeragetNotesStrategies + '.pkl')
    if os.path.isfile(pretrained_data_path):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', pretrained_data_path)
        with open(pretrained_data_path, 'rb') as f:
            (data, names, discretizer, normalizer) = pickle.load(f)
    else:
        # If we did not already processed the data we do it here
        reader_dir = os.path.join(CFG.ihm_dataset_dir, 'train' if (mode == 'train') or mode == 'val' else 'test')
        listfile_path = os.path.join(reader_dir, 'listfile.csv')
        reader = InHospitalMortalityReader(
            dataset_dir=reader_dir,
            listfile=listfile_path,
            period_length=48.0)
        
        if normalizer is None:
            discretizer = Discretizer(
                timestep=float(CFG.timestep),
                store_masks=True,
                impute_strategy='previous',
                start_time='zero')
        
            discretizer_header = discretizer.transform(reader.read_example(0)["X"])[1].split(',')
            cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
            
            # text reader for reading the texts
            if (mode == 'train') or (mode == 'val'):
                text_reader = utils.TextReader(CFG.textdataset_fixed_dir, CFG.starttime_path)
            else:
                text_reader = utils.TextReader(CFG.test_textdataset_fixed_dir, CFG.test_starttime_path)
            
            # choose here which columns to standardize
            normalizer = Normalizer(fields=cont_channels)
            normalizer_state = CFG.normalizer_state
            if normalizer_state is None:
                normalizer_state = f'ihm_ts{CFG.timestep}.input_str:{CFG.imputation}.start_time:zero.normalizer'
                normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
            normalizer.load_params(normalizer_state)

        normalizer = None
        # Load the patient data
        train_raw = ihm_utils.load_data(reader, discretizer, normalizer, CFG.small_part, return_names=True)
        
        print("Number of train_raw_names: ", len(train_raw['names']))
        
        data = concat_text_timeseries(text_reader, train_raw)
        
        train_names = list(data[3])
        
        os.makedirs(CFG.pretrained_data_dir, exist_ok=True)
        with open(pretrained_data_path, 'wb') as f:
            # Write the processed data to pickle file so it is faster to just read later
            pickle.dump((data, train_names, discretizer, normalizer), f)
        
    data_X = data[0]        # (14130, 48, 76)
    data_y = data[1]        # (14130,)
    data_text = data[2]     # (14130, n_times, note_length)
    data_names = data[3]
    data_times = data[4]    # (14130, n_times)
    start_times = data[5]   # (14130), start time in data_times
    timetoends = get_time_to_end_diffs(data_times, start_times) # (14130, n_times), from largest to smallest
    # start_time: first charttime
    # data_times: time from clinical notes. 
    # timetoends: hours, how long to the end time (starttime + 49). 
    # Notes that first time in 'data_times' might be smaller than start_time, which means the clinical notes time is earlier than charttime. 
    # all ( time-stamps - start-time ) <= 48h, but timetoends can be larger than 48 since the notes time can be earlier than charttime. 

    # for note length in every time stamp for all patients, avg length 228.68, std 173.16
    if ARGS['model_name'] == 'BioBert':
        tokenized_path = os.path.join(CFG.pretrained_data_dir, EmbedModelName + '_tokenized_ids_attns_' + mode + '_' + AggeragetNotesStrategies + '_' + str(ARGS['MaxLen']) + '_truncate.pkl')
        if os.path.isfile(tokenized_path):
            # If the pickle file containing text_ids exists we will just load it and save time by not computing it using the tokenizer
            with open(tokenized_path, 'rb') as f:
                txt_ids, attention_masks = pickle.load(f)
            print('txt_ids, attention_masks are loaded from Tokenize ', tokenized_path)
        else:
            txt_ids, attention_masks = tokenizeGetIDs(tokenizer, data_text, ARGS['MaxLen'])
            with open(tokenized_path, 'wb') as f:
                # Write the output of tokenizer to a pickle file so we can use it later
                pickle.dump((txt_ids, attention_masks), f)
            print('txt_ids, attention_masks is written to Tokenize ', tokenized_path)
    else:
        txt_ids = data_text.copy()
        attention_masks = list(np.ones_like(txt_ids))

    # Remove the data when text is empty
    indices = []
    for idx, txt_id in enumerate(txt_ids):
        # txt_ids: (n_patient, n_times, max_len)
        if len(txt_id) == 0:
            indices.append(idx)
        else:
            if ARGS['NumOfNotes'] > 0:
                # Only pick the last note
                txt_ids[idx] = txt_id[-ARGS['NumOfNotes']:]
                if ARGS['model_name'] == 'BioBert':
                    attention_masks[idx] = attention_masks[idx][-ARGS['NumOfNotes']:]
                
    for idx in reversed(indices):
        txt_ids.pop(idx)
        attention_masks.pop(idx)
        data_X = np.delete(data_X, idx, 0)
        data_y = np.delete(data_y, idx, 0)

        data_text.pop(idx)
        data_times.pop(idx)
        timetoends.pop(idx)
        data_names = np.delete(data_names, idx, 0)
        start_times = np.delete(start_times, idx, 0)
    del data
    
    return txt_ids, attention_masks, data_X, data_y, data_text, data_names, timetoends, discretizer, normalizer


def generate_tensor_text(t, w2i_lookup, MaxLen):
    # Tokenize for Clinical notes model
    t_new = []
    max_len = -1
    for text in t:
        tokens = list(map(lambda x: lookup(w2i_lookup, x), str(text).split()))
        if MaxLen > 0:
            tokens = tokens[:MaxLen]
        t_new.append(tokens)
        max_len = max(max_len, len(tokens))
    pad_token = w2i_lookup['<pad>']
    for i in range(len(t_new)):
        if len(t_new[i]) < max_len:
            t_new[i] += [pad_token] * (max_len - len(t_new[i]))
    return np.array(t_new)


def generate_padded_batches(x, y, text, text_ids, data_attn, data_times, batch_size, w2i_lookup):
    # Generate batches
    batches = []
    begin = 0
    while begin < len(text):            
        end = min(begin+batch_size, len(text))
        x_slice = np.stack(x[begin:end])
        y_slice = np.stack(y[begin:end])
        if TXT_MODEL_TYPE == 'BioBert':
            tensor_batch = [torch.tensor(seq, dtype=torch.long) for seq in text_ids[begin:end]]
            text_tensor = pad_sequence(tensor_batch, batch_first=True, padding_value=0)  # shape: (B, T_max)
            text_slice = text_tensor.numpy()  # NumPy로 변환할 경우만 (모델에 바로 쓸 땐 텐서 그대로)
            # text_slice = np.array(text_ids[begin:end])
        else:
            tensor_batch = [torch.tensor(seq, dtype=torch.long) for seq in text_ids[begin:end]]
            text_tensor = pad_sequence(tensor_batch, batch_first=True, padding_value=0)  # shape: (B, T_max)
            text_slice = generate_tensor_text(text_ids[begin:end], w2i_lookup, ARGS['MaxLen'])

        attn_batch = [torch.tensor(mask, dtype=torch.long) for mask in data_attn[begin:end]]
        attn_tensor = pad_sequence(attn_batch, batch_first=True, padding_value=0)
        attn_slice = attn_tensor.numpy()
        # attn_slice = np.array(data_attn[begin:end])

        time_batch = [torch.tensor(time, dtype=torch.long) for time in data_times[begin:end]]
        time_tensor = pad_sequence(time_batch, batch_first=True, padding_value=0)
        time_slice = time_tensor.numpy()
        # time_slice = np.array(data_times[begin:end], dtype=object)
        batches.append((x_slice, y_slice, text_slice, attn_slice, time_slice))
        begin += batch_size
    return batches


def validate(model, data_X_val, data_y_val, data_text_val, txt_ids_eval, attn_val, times_val, names_eval, batch_size, word2index_lookup,
             last_best_val_aucpr, save):
    val_batches = generate_padded_batches(
        data_X_val, data_y_val, data_text_val, txt_ids_eval, attn_val, times_val, batch_size, word2index_lookup)
    
    prf_val, probablities = Evaluate_Model(model, val_batches, names_eval)
    loss_value = prf_val['epoch_loss']
        
    final_aucroc = prf_val['roc_macro']
    final_aucpr = prf_val['aucpr']

    final_acc = prf_val['accuracy']
    final_recall = prf_val['recall']
    final_precision = prf_val['precision']
    final_f1 = prf_val['fscore']
    final_sensitivity = prf_val['sensitivity']
    final_sprcificity = prf_val['specificity']

    print("Validation Loss: {:.4f}, AUCPR: {:.4f}, AUCROC: {:.4f}, accuracy {:.4f}, recall {:.4f}, precision {:.4f}, f1 {:.4f}, sensitivity {:.4f}, specificity {:.4f}".format(loss_value, final_aucpr, final_aucroc, final_acc, final_recall, final_precision, final_f1, final_sensitivity, final_sprcificity))

    log_metrics(writer, f"{MODEL_NAME_OUT}/{MODE}", 0, prf_val)
    # writer.add_pr_curve('Test/PR_Curve', labels, preds, global_step=0)
    
    changed = False
    if final_aucpr > last_best_val_aucpr:
        changed = True
        if save:
            save_path = os.path.join(CFG.pretrained_checkpoints_dir, MODEL_NAME_OUT)
            torch.save({'state_dict': model.state_dict()}, save_path)
            print("Best Model saved in", save_path)
    return max(last_best_val_aucpr, final_aucpr), changed, probablities, final_aucroc, prf_val


def write_probs(PatientNames, test_data_text, probs, test_data_y, path):
    df = pd.DataFrame({
        'names': PatientNames,
        'text': test_data_text,
        'probs': probs.detach().cpu(),
        'Label': test_data_y})
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path,index=False)

start_time = time.time()
# Ignoring warnings
warnings.filterwarnings('ignore')

CFG = utils.get_config() # Get configs
ARGS = utils.get_args() # Get arguments

os.environ["CUDA_VISIBLE_DEVICES"] = ARGS['gpu_id']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if ARGS["Seed"]:
    torch.manual_seed(ARGS["Seed"])
    np.random.seed(ARGS["Seed"])

## Loading pre-trained model based on EmbedModel argument
if ARGS['model_name'] == 'BioBert':
    if ARGS["EmbedModel"] == "bioRoberta":
        EmbedModelName = "bioRoberta"
        tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
        BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")
    elif ARGS["EmbedModel"] == "BioBert":
        EmbedModelName = "BioBert"
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        BioBert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif ARGS["EmbedModel"] == "Bert":
        EmbedModelName = "SimpleBert"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BioBert = BertModel.from_pretrained("bert-base-uncased")
    elif ARGS["EmbedModel"] == "MedBert":
        EmbedModelName = "MedBert"
        tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        BioBert = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    BioBert = BioBert.to(device)
    BioBertConfig = BioBert.config
    # conf.max_len = BioBertConfig.max_position_embeddings
    W_emb = None
    lookup = None
    word2index_lookup = None
    


print(str(vars(CFG)))
print(str(ARGS))



text_model_checkpoint_path = os.path.join(CFG.pretrained_checkpoints_dir, ARGS['TextModelCheckpoint'])

# Use text model checkpoint to update the bert model to fine-tuned weights
if (ARGS['model_name'] == 'BioBert') and (ARGS['TextModelCheckpoint'] != None):
    ## BioClinicalBERT_FT
    checkpoint = torch.load(
        text_model_checkpoint_path, 
        weights_only=False,
        map_location=device
    ) 
    BioBert = checkpoint.BioBert
    del checkpoint

if (ARGS['model_name'] == 'BioBert') and bool(int(ARGS['freeze_model'])):
    for param in BioBert.parameters():
        param.requires_grad = False

AggeragetNotesStrategies = 'Mean'
txt_ids, attention_masks, data_X, data_y, data_text, data_names, data_times, discretizer, normalizer = \
    Read_Aggregate_data('train', AggeragetNotesStrategies, discretizer=None, normalizer = None)

txt_ids_eval, attention_masks_eval, eval_data_X, eval_data_y, eval_data_text, eval_data_names, eval_data_times, _, _ = \
    Read_Aggregate_data('val', AggeragetNotesStrategies, discretizer=discretizer, normalizer = normalizer)

txt_ids_test, attention_masks_test, test_data_X, test_data_y, test_data_text, test_data_names, test_data_times, _, _ = \
    Read_Aggregate_data('test', AggeragetNotesStrategies, discretizer=discretizer, normalizer = normalizer)

# len(data_X)         # 14068
# len(eval_data_X)    # 3086
# len(test_data_X)    # 3107
# np.sum(data_y==1)1852; np.sum(data_y==0) 12216
# np.sum(eval_data_y==1), np.sum(eval_data_y==0) (404, 2682)
# np.sum(test_data_y==1), np.sum(test_data_y==0)  (359, 2748)


# 10.247512084162638 7.322470794860153
# 228.68260013040884 173.16657884505082

# 10.098185353208036 5.9940900316217
# 229.1103231396207 172.51297922604402

# 10.149983907306083 7.189111932829197
# 227.58609208523592 171.34610446282736


def log_metrics(writer, title, step, prf_val):
    writer.add_scalar(f"{title}/Loss", prf_val['epoch_loss'], step)
    writer.add_scalar(f"{title}/AUCROC", prf_val['roc_macro'], step)
    writer.add_scalar(f"{title}/AUCPR", prf_val['aucpr'], step)
    writer.add_scalar(f"{title}/Accuracy", prf_val['accuracy'], step)
    writer.add_scalar(f"{title}/Recall", prf_val['recall'], step)
    writer.add_scalar(f"{title}/Precision", prf_val['precision'], step)
    writer.add_scalar(f"{title}/F1", prf_val['fscore'], step)


def train_loop(model, epochs, learning_rate):
    early_stopping = 0
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)# decay, dynamic
    print("--- Loaded everything %s seconds ---" % (time.time() - start_time))
    global data_X, data_y, data_text, txt_ids, attention_masks, data_times
    global eval_data_X, eval_data_y, eval_data_text, txt_ids_eval, attention_masks_eval, eval_data_times
    global test_data_X, test_data_y, test_data_text, txt_ids_test, attention_masks_test, test_data_times

    log_path = os.path.join(CFG.log_dir, 'ManLogs', MODEL_NAME_OUT + '_ManLog.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a+') as f:
        print("--- Loaded everything %s seconds ---" % (time.time() - start_time), file=f)


    for epoch in range(epochs):
        print("Starting training for epoch: %d" % epoch)

        data = list(zip(data_X, data_y, data_text, txt_ids, attention_masks, data_times))
        random.shuffle(data)
        data_X, data_y, data_text, txt_ids, attention_masks, data_times = zip(*data)
        del data

        print("Preparing batches for the epoch!")
        batches = generate_padded_batches(
            data_X, data_y, 
            data_text, txt_ids, attention_masks, data_times, 
            BATCH_SIZE, word2index_lookup)

        print("Start train for the train step!")
        prf_train = train_step(model, batches, epoch, epochs, 50, optimizer)

        aucroc_value = prf_train['roc_macro']
        loss_value = prf_train['epoch_loss']
        current_aucroc = aucroc_value
        
        print("Loss: %f - AUCROC: %f" %(loss_value, current_aucroc))
        writer.add_scalar(f"{MODEL_NAME_OUT}/Train/Loss", loss_value, epoch)
        writer.add_scalar(f"{MODEL_NAME_OUT}/Train/AUCROC", current_aucroc, epoch)

        del batches

        print("Start Evaluation for current epoch : %d" % epoch)
        last_best_val_aucpr = -1
        last_best_val_aucpr, changed, probs, last_best_val_auc, prf_val = validate(
            model, eval_data_X, eval_data_y, eval_data_text, txt_ids_eval, attention_masks_eval, eval_data_times,
            eval_data_names, BATCH_SIZE, word2index_lookup, last_best_val_aucpr, save=True)

        log_metrics(writer, f"{MODEL_NAME_OUT}/Train/Val", epoch, prf_val)

        if changed == False:
            early_stopping += 1
            print("Didn't improve!: " + str(early_stopping))
        else:
            early_stopping = 0

        if early_stopping >= 15:
            print("AUCPR didn't change from last 15 epochs, early stopping")
            break

        print("*End of Epoch.*\n")


def test_loop(model, checkpoint_path, mode='val'):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False)
    
    model.load_state_dict(checkpoint['state_dict'])

    last_best_val_aucpr = -1
    if mode == 'val':
        print("Validating the model on validation set")
        last_best_val_aucpr, _, probs, last_best_val_auc, prf_val = validate(
            model, eval_data_X, eval_data_y, eval_data_text, txt_ids_eval, attention_masks_eval, 
            eval_data_times, eval_data_names, BATCH_SIZE, word2index_lookup, last_best_val_aucpr, False)
    elif mode == 'test':
        print("Testing the model on test set")
        last_best_val_aucpr, _, probs, last_best_val_auc, prf_val = validate(
            model, test_data_X, test_data_y, test_data_text, txt_ids_test, attention_masks_test, 
            test_data_times, test_data_names, BATCH_SIZE, word2index_lookup, last_best_val_aucpr, False)
    
    output_file_path = os.path.join(CFG.output_dir, MODEL_NAME_OUT + '.txt')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'a+') as f:
        print("Val AUC : ", last_best_val_auc, file=f)
        print(f"TEST, AUCPR: {prf_val['aucpr']:.4f}",
              f"AUCROC: {prf_val['roc_macro']:.4f}",
              f"accuracy {prf_val['accuracy']:.4f}",
              f"recall {prf_val['recall']:.4f}",
              f"precision {prf_val['precision']:.4f}",
              f"f1 {prf_val['fscore']:.4f}",
              f"sensitivity {prf_val['sensitivity']:.4f}",
              f"specificity {prf_val['specificity']:.4f}", file=f)


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=os.path.join(CFG.log_dir, 'Summary'))
models = [Models.mm_transformer]
# models = [Models.mm_transformer, Models.mm_transformer_2, Models.mm_transformer_1]

EPOCHS = int(ARGS['number_epoch'])
BATCH_SIZE = int(ARGS['batch_size'])
LR = float(ARGS['learning_rate'])  # Learning rate for the optimizer

TXT_MODEL_TYPE = ARGS['model_name']
TS_MODEL_TYPE = ARGS['TSModel']  # 'LSTM', 'GRU', 'Transformer'
MODEL_USAGE = ARGS['model_type'] # 'both'
MODE = ARGS['mode']  # 'train', 'val', 'test'

for model_type in models:
    # reassign model name path
    MODEL_NAME_OUT = model_type.__name__
    model = model_type(
        txt_model_type = TXT_MODEL_TYPE,
        model_usage = MODEL_USAGE, 
        BioBert = BioBert, 
        ts_model_type = TS_MODEL_TYPE, 
        device = device
    ).to(device)
    
    if MODE == 'train':
        train_loop(model, EPOCHS, LR)
    else:
        checkpoint_path = os.path.join(CFG.pretrained_checkpoints_dir, MODEL_NAME_OUT)
        test_loop(model, checkpoint_path, mode=MODE)
        
writer.close()