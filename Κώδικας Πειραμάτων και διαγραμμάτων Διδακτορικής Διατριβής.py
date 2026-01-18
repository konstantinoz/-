#Κώδικας Πειραμάτων και διαγραμμάτων Διδακτορικής Διατριβής
#Zero – shot classification (updated με LIME – SHAP και Reliability diagrams)
# Αποσύνδεση της προεγκατεστημένης cache του Google Colab
!pip uninstall -y datasets
!pip uninstall -y huggingface_hub
!pip uninstall -y transformers

# Επαναφορά εξόδου για καθαρή οθόνη
from IPython.display import clear_output
clear_output()
print("Προεγκατεστημένα πακέτα διαγράφηκαν. Επαναλαμβάνουμε την εγκατάσταση.")

# Εγκατάσταση των σωστών εκδόσεων από το Hugging Face
!pip install datasets==3.0.0 huggingface_hub transformers
clear_output()
print("Εγκαταστάθηκαν ξανά οι βιβλιοθήκες. Κάνε restart το runtime και ξανατρέξε.")

from IPython.display import clear_output
!pip install lime --quiet
!pip install shap --quiet
clear_output()

import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import shap
from lime.lime_text import LimeTextExplainer
from datasets import load_dataset
import datasets
print(f"Datasets module path: {datasets.__file__}")
print(f"Datasets version: {datasets.__version__}")

import os  # For interacting with the operating system
import torch  # For GPU/CPU handling and tensor computations
import textwrap  # For formatting text output

from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer
)  # Hugging Face Transformers for loading models and tokenizers

# Set device: use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose a zero-shot classification model
model_name = "valhalla/distilbart-mnli-12-1"

# Load pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize zero-shot classification pipeline
classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1
)

# Φόρτωση του dataset
Data1 = load_dataset(sudhanshusinghaiml/airlines-sentiment-analysis') 
# 0: negative, 1: neutral, 2: positive
#clear_output()
df = pd.DataFrame(data1['train'])
df = df.rename(columns={'label': 'target'})
df = df[df['text'].notna()]

X = df['text']
y = df['target']

print(df.columns)

import torch

# Define candidate labels explicitly
candidate_labels = ["negative", "neutral", "positive"]
label2idx = {v: k for k, v in enumerate(candidate_labels)}

# Prepare containers
probs, preds, lengths = [], [], []

# Loop over all texts
texts = df['text'].tolist()
true_labels = df['target'].tolist()

for i, text in tqdm(enumerate(texts), total=len(texts)):
    result = classifier(text, candidate_labels=candidate_labels)

    # Get scores in order of candidate_labels
    score_dict = {label: score for label, score in zip(result["labels"], result["scores"])}
    prob_row = [score_dict[label] for label in candidate_labels]

    probs.append(prob_row)
    preds.append(label2idx[result['labels'][0]])# Most probable label index
    lengths.append(len(text.split())) # Length in tokens or words

len(df)

N = len(true_labels)

df_preds = pd.DataFrame({
    'negative': [p[0] for p in probs[:N]],
    'neutral':  [p[1] for p in probs[:N]],
    'positive': [p[2] for p in probs[:N]],
    'predicted_labels': preds[:N],
    'target': true_labels,
    'length': lengths[:N]
})

# Save to CSV
df_preds.to_csv("[distilbart-mnli-12-1_zeroshot_financial]__TestPredictions.csv", index=False)

#from lime.lime_text import LimeTextExplainer

class_names = ["negative", "neutral", "positive"]

def predict_proba(texts):
    result = []
    for text in texts:
        output = classifier(text, candidate_labels=class_names)
        result.append(output['scores'])
    return np.array(result)

#from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=["negative", "neutral", "positive"])

# Γραμμές που θέλεις να εξηγήσουμε
indexes_to_explain = [39, 44, 128, 131]

for idx in indexes_to_explain:
    print(f"\nߔ͠LIME explanation for index {idx}:\n")
    print("ߓݠText:", X.iloc[idx])
    exp = explainer.explain_instance(X.iloc[idx], predict_proba, num_features=10)
    exp.show_in_notebook(text=True)

#import shap

#tokenizer από το pipeline
tokenizer = classifier.tokenizer

# SHAP Text masker με Hugging Face tokenizer
masker = shap.maskers.Text(tokenizer)

# Φτιάξε τον εξηγητή
explainer = shap.Explainer(predict_proba, masker=masker, output_names=["negative", "neutral", "positive"])

# Επιλογή κειμένων
indexes_to_explain = [39, 44, 128, 131]
texts_to_explain = [X.iloc[i] for i in indexes_to_explain]

# Υπολογισμός SHAP values
shap_values = explainer(texts_to_explain)

# Πλοκή για κάθε instance
for i in range(len(indexes_to_explain)):
    print(f"\nߔΠSHAP Explanation for index {indexes_to_explain[i]}:\n")
    shap.plots.text(shap_values[i])

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, cohen_kappa_score

# Πάρε τις προβλέψεις και τους πραγματικούς στόχους
y_pred = df_preds['predicted_labels']
y_true = df_preds['target']

# Υπολογισμός μετρικών
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

# Εκτύπωση αποτελεσμάτων
print("ߓʠEvaluation Metrics:")
print("Accuracy:", round(accuracy, 3))
print("F1 (macro):", round(f1, 3))
print("Precision (macro):", round(precision, 3))
print("Recall (macro):", round(recall, 3))
print("Matthews Correlation Coefficient:", round(mcc, 3))
print("Cohen's Kappa:", round(kappa, 3))

import seaborn as sns

# Confusion matrix (normalized)
cm = confusion_matrix(y_true, y_pred, normalize='true')

def plot_cm(cm, model_name, dpi=300):
    classes = ['negative', 'neutral', 'positive']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(8, 6), dpi=dpi)
    ax = sns.heatmap(df_cm, annot=True, fmt='.3f', cbar=True, annot_kws={"size": 11}, cmap='Blues')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(model_name)
    plt.show()

plot_cm(cm, model_name, dpi=300)
print("Confusion Matrix (normalized):")
print(cm)

df_preds['confidence'] = df_preds[['negative', 'neutral', 'positive']].max(axis=1)
df_preds['correct'] = df_preds['predicted_labels'] == df_preds['target']
print("Avg confidence correct:", df_preds[df_preds['correct']]['confidence'].mean())
print("Avg confidence incorrect:", df_preds[~df_preds['correct']]['confidence'].mean())

prob_true, prob_pred = calibration_curve(df_preds['correct'], df_preds['confidence'], n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('Reliability Diagram')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.show()
plt.savefig(f"reliability_{model_name.replace('/', '_')}_zeroshot_financial.png", dpi=300)
plt.close()

# Fine tuning Sentiment analysis και Reliability diagrams 
'''
!pip install torchinfo
!pip install transformers
!pip install datasets
!pip install sentencepiece
'''

# Εισαγωγή απαραίτητων βιβλιοθηκών
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
from transformers import AdamW
from torch.optim import Adam
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)
import time
import torch
from sklearn.model_selection import train_test_split
from torchinfo import summary
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers import pipeline
import re
from datasets import load_dataset, Dataset, DatasetDict
import os
import io
import tempfile
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset, DatasetDict
from scipy.special import softmax
from tqdm import tqdm
from IPython.display import clear_output
import time

# Φόρτωση έτοιμου dataset από το HuggingFace Hub
data1 = load_dataset('Sp1786/multiclass-sentiment-analysis-dataset')
df1_train = pd.DataFrame(data1['train'])
df1_test = pd.DataFrame(data1['test'])
df1_val = pd.DataFrame(data1['validation'])
from IPython.display import clear_output
clear_output()

# Ενοποίηση των συνόλων δεδομένων σε ένα DataFrame
df = pd.concat([df1_train, df1_test, df1_val], ignore_index=True)

# Δημιουργία λεξικού για τις κατηγορίες
mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Μετονομασία της στήλης label σε target
df = df.rename(columns={'label': 'target'})

# Εκτύπωση κατανομής των κλάσεων
df['target'].value_counts()
df.head()

# Υπολογισμός αριθμού instances ανά κατηγορία για visualization
data = df['target'].value_counts().reset_index()
data.columns = ['Classes', 'Count']
n_colors = len(data)
colors = sns.color_palette("colorblind", n_colors)
cmap = sns.color_palette(colors, as_cmap=True)

# Δημιουργία barplot για την κατανομή
plt.figure(dpi=300)
ax = sns.barplot(x='Classes', y='Count', data=data, hue='Classes',
                 palette=cmap, legend=False)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
     (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10),
                textcoords='offset points')
plt.title('Sp1786/multiclass-sentiment-analysis')
plt.xlabel('Classes')
plt.ylabel('Number of Instances')
plt.tight_layout()
sns.despine(right=True, top=True)
plt.show()

# Καθαρισμός κειμένων (προεπεξεργασία)
import re
from nltk.corpus import stopwords
import string

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in
                     stopwords.words('english')])

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def add_space_around_punctuation(text):
    text = re.sub(r'\b(\w+)\b', r' \1 ', text)
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    return text

def remove_punct(text):
    text = add_space_around_punctuation(text)
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_username(text):
    return re.sub('@[^\s]+', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def decontraction(text):
    text = re.sub(r"won\'t", " will not", text)
    return text

def seperate_alphanumeric(text):
    words = re.findall(r"[^\W\d_]+|\d+", text)
    return " ".join(words)

def char(text):
    return re.sub(r'[^a-zA-Z]', ' ', text)

# Εφαρμογή όλων των παραπάνω καθαρισμών
df['final_text'] = df['text'].fillna('')
df['final_text'] = df['final_text'].apply(lambda x: str(x))
df['final_text'] = df['final_text'].apply(lambda x: remove_username(x))
df['final_text'] = df['final_text'].apply(lambda x: remove_url(x))
df['final_text'] = df['final_text'].apply(lambda x: remove_emoji(x))
df['final_text'] = df['final_text'].apply(lambda x: decontraction(x))
df['final_text'] = df['final_text'].apply(
    lambda x: seperate_alphanumeric(x))
df['final_text'] = df['final_text'].apply(lambda x: char(x))
df['final_text'] = df['final_text'].apply(lambda x: x.lower())

# Συνάρτηση για έλεγχο κενών τιμών
def check_empty_and_print_length(df, column_name):
    empty_count = df[column_name].apply(lambda x: x.strip() == '').sum()
    print(f"Number of empty entries in {column_name}: {empty_count}")
    print(f"Length of DataFrame: {len(df)}")

# Συνάρτηση για αφαίρεση κενών τιμών
def remove_empty_rows_and_print_length(df, column_name):
    df = df[df[column_name].apply(lambda x: x.strip() != '')]
    print(f"Length of DataFrame after removing empty rows: {len(df)}")
    return df

# Έλεγχος κενών
target_col = 'final_text'
check_empty_and_print_length(df, target_col)
df = remove_empty_rows_and_print_length(df, target_col)

# Ορισμός χαρακτηριστικών και ετικετών
X = df['final_text']
y = df['target']

empty_count = df['text'].isna().sum()
print(f"Number of empty strings in 'text': {empty_count}")

# Διπλό split για 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Δημιουργία DataFrames για κάθε split
train_data = pd.DataFrame({'text': X_train, 'label': y_train})
test_data = pd.DataFrame({'text': X_test, 'label': y_test})
val_data = pd.DataFrame({'text': X_val, 'label': y_val})

# Δημιουργία DatasetDict
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_data),
    'test': Dataset.from_pandas(test_data),
    'validation': Dataset.from_pandas(val_data)
})

# Ρύθμιση token και μοντέλου
os.environ['HF_TOKEN'] = "*****" 
model_name = "EleutherAI/pythia-410m"
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          use_auth_token=os.getenv(
                                              'HF_TOKEN'))
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,use_auth_token=os.getenv('HF_TOKEN'))
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenization
max_length = 64
def tokenize_function(examples):
    return tokenizer(examples['text'], 
                     padding='max_length', truncation=True,
                     max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Συνάρτηση υπολογισμού metrics
def comp_metrics(eval_pred):
    if isinstance(eval_pred.predictions, tuple):
        logits = eval_pred.predictions[0]
    else:
        logits = eval_pred.predictions
    labels = eval_pred.label_ids
    probabilities = softmax(logits, axis=-1)
    predictions = np.argmax(probabilities, axis=-1)
    predictions = predictions.flatten()
    labels = labels.flatten()
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1, 
            'precision': precision, 'recall': recall}

# Grid για hyperparameter tuning
param_grid = {
    'num_train_epochs': [3, 5, 7, 10],
    'per_device_train_batch_size': [16, 32],
    'learning_rate': [5e-5, 3e-5, 2e-5]
}

best_metrics = None
best_params = None
start_time = time.time()
torch.utils.checkpoint.use_reentrant = True

# Βρόχος για grid search
for num_train_epochs in tqdm(param_grid['num_train_epochs'],
                             desc="Training epochs"):
    for per_device_train_batch_size in tqdm(
        param_grid['per_device_train_batch_size'],
                                            desc="Batch sizes", 
        leave=False):
        for learning_rate in tqdm(param_grid['learning_rate'],
                                  desc="Learning rates", leave=False):
            print(f"Training with num_train_epochs={num_train_epochs},
            per_device_train_batch_size={per_device_train_batch_size},
                  learning_rate={learning_rate}")
            model.config.use_cache = False
            training_args = TrainingArguments(
                output_dir='training_dir',
                evaluation_strategy='epoch',
                save_strategy='epoch',
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                learning_rate=learning_rate,
                logging_dir='./logs',
                load_best_model_at_end=True,
                metric_for_best_model='accuracy',
                save_total_limit=1,
                gradient_accumulation_steps=16,
                gradient_checkpointing=True,
                fp16=True
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                # Χρησιμοποιούμε validation για αξιολόγηση
                tokenizer=tokenizer,
                compute_metrics=comp_metrics,
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=2)]
            )
            trainer.train()
            eval_metrics = trainer.evaluate()
            print(f"Evaluation metrics: {eval_metrics}")
            if best_metrics is None or
                  eval_metrics['eval_accuracy'] > 
                  best_metrics['eval_accuracy']:
                best_metrics = eval_metrics
                best_params = {
                    'num_train_epochs': num_train_epochs,
                    'per_device_train_batch_size': 
                    per_device_train_batch_size,
                    'learning_rate': learning_rate
                }

# Εκτύπωση καλύτερων παραμέτρων και metrics
print("Best parameters found:")
print(best_params)
print("Best evaluation metrics:")
print(best_metrics)

end_time = time.time()
execution_time = end_time - start_time
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = execution_time % 60
print(
    f"Execution time: {hours} hours, 
    {minutes} minutes, {seconds:.2f} seconds")

# Αποθήκευση καλύτερου μοντέλου
savedmodel = pipeline('text-classification', model=trainer.model,
                      tokenizer=tokenizer, device=0)
test_labels = y_test

# Υπολογισμός προβλέψεων και πιθανοτήτων
labels_dict = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2}
def count_words(text):
    return len(text.split())
probs = []
preds = []
lengths = []
for text in tqdm(X_test.tolist(), desc="Processing", unit="text"):
    pred = savedmodel(text, return_all_scores=True)
    prob = [0, 0, 0]
    for p in pred[0]:
        label_idx = labels_dict[p['label']]
        prob[label_idx] = p['score']
    probs.append(prob)
    preds.append(prob.index(max(prob)))
    lengths.append(count_words(text))

# DataFrame με αποτελέσματα
predsVSlabels = pd.DataFrame({
    'negative': [p[0] for p in probs],
    'neutral': [p[1] for p in probs],
    'positive': [p[2] for p in probs],
    'predicted_labels': preds,
    'target': y_test.tolist(),
    'length': lengths
})
pd.options.display.float_format = '{:.6f}'.format
predsVSlabels.to_csv(
    "[pythia-base_ADAM]__TestPredictions.csv", index=True)

# Confusion Matrix
cm = confusion_matrix(y_test, preds, normalize='true')
from sklearn.utils.multiclass import unique_labels
classes = unique_labels(y_test, preds)
print("Σειρά κλάσεων:", classes)

def plot_cm(cm, model_name, dpi=300):
    classes = ['negative', 'neutral', 'positive']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(8, 6), dpi=dpi)
    ax = sns.heatmap(df_cm, annot=True, fmt='.3f', cbar=True,
                     annot_kws={"size": 11}, cmap='Blues')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(model_name)
    plt.show()
plot_cm(cm, model_name, dpi=300)

# ROC AUC Scores
y_pred_probs = predsVSlabels[['negative', 'neutral', 
                              'positive']].values
y_true = predsVSlabels['target'].values
roc_auc_ovr = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
roc_auc_ovo = roc_auc_score(y_true, y_pred_probs, multi_class='ovo')
print(f"ROC AUC (One-vs-Rest): {roc_auc_ovr}")
print(f"ROC AUC (One-vs-One): {roc_auc_ovo}")

# Μετρικές αξιολόγησης
predicts = predsVSlabels['predicted_labels']
accuracy = accuracy_score(y_true, predicts)
weighted_f1_score = f1_score(y_true, predicts, average='weighted')
macro_avg_f1_score = f1_score(y_true, predicts, average='macro')
general_f1_score = f1_score(y_true, predicts, average='micro')
f1_per_class = f1_score(y_true, predicts, average=None)
mcc = matthews_corrcoef(y_true, predicts)
cohen_kappa = cohen_kappa_score(y_true, predicts)
precision_micro = precision_score(y_true, predicts, average='micro')
precision_macro = precision_score(y_true, predicts, average='macro')
recall_micro = recall_score(y_true, predicts, average='micro')
recall_macro = recall_score(y_true, predicts, average='macro')
precision_per_class = precision_score(y_true, predicts, average=None)
recall_per_class = recall_score(y_true, predicts, average=None)

# Εκτύπωση μετρικών
print("model_name =", model_name)
print("Accuracy Score:", accuracy)
print("Weighted F1 Score:", weighted_f1_score)
print("ROC AUC Score (OVR):", roc_auc_ovr)
print("ROC AUC Score (OVO):", roc_auc_ovo)
print("Matthews Correlation Coefficient:", mcc)
print("Cohen’s Kappa Coefficient:", cohen_kappa)
print("F1 Score per Class:

# Εκτύπωση F1 ανά κλάση
class_names = ['Negative', 'Neutral', 'Positive']
for name, score in zip(class_names, f1_per_class):
    print(f"{name}: {score}")
print("Macro-Averaged F1 Score:", macro_avg_f1_score)
print("General F1 Score:", general_f1_score)
print("Micro-averaged F1 score:", general_f1_score)
print("Precision (Micro):", precision_micro)
print("Precision (Macro):", precision_macro)
print("Recall (Micro):", recall_micro)
print("Recall (Macro):", recall_macro)
print("Precision per class:")
for name, score in zip(class_names, precision_per_class):
    print(f"{name}: {score}")
print("Recall per class:")
for name, score in zip(class_names, recall_per_class):
    print(f"{name}: {score}")

# Εκτύπωση δομής μοντέλου
summary(model)

# Υπολογισμός calibration metrics
df = predsVSlabels
df['confidence'] = df[['negative', 'neutral', 'positive']].max(axis=1)
df['correct'] = df['predicted_labels'] == df['target']
mean_confidence_correct = df[df['correct']]['confidence'].mean()
mean_confidence_incorrect = df[~df['correct']]['confidence'].mean()
print(f"Μέση εμπιστοσύνη για σωστές προβλέψεις: {
    mean_confidence_correct:.4f}")
print(
    f"Μέση εμπιστοσύνη για λανθασμένες προβλέψεις: 
    {mean_confidence_incorrect:.4f}")

# Καμπύλη αξιοπιστίας
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(df['correct'], df['confidence'], 
                                         n_bins=10)
plt.figure(dpi=600, figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Model Reliability')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.title(f'Reliability Diagram of {model_name}')
plt.xlabel('Average Predicted Confidence')
plt.ylabel('True Accuracy')
plt.legend()
plt.show()

# Υπολογισμός Expected Calibration Error (ECE)
def expected_calibration_error(confidences, corrects, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = corrects[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(
                avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

ece = expected_calibration_error(df['confidence'], df['correct'])
print(f"Expected Calibration Error (ECE): {ece:.4f}")


#Γεωμετρική αναπαράσταση τρισδιάστατου Vector Space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define 3D coordinates for each word embedding
embeddings = {
    "Βασιλιάς": np.array([2.0, 3.2, 5.0]),
    "Άνδρας":   np.array([2.5, 2.8, 4.2]),
    "Φάλαινα":  np.array([-2.0, -3.0, 0.5]),
    "Δελφίνι":  np.array([-2.4, -3.5, 0.9]),
}

# Prepare figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the embeddings
for word, vec in embeddings.items():
    ax.scatter(*vec, color='red', s=100, marker='x')# Bigger X markers
    ax.text(*vec, word, fontsize=12, 
            #weight='bold'
            )

# Draw dashed lines to connect similar pairs
pairs = [("Βασιλιάς", "Άνδρας"), ("Φάλαινα", "Δελφίνι")]
for w1, w2 in pairs:
    x_vals = [embeddings[w1][0], embeddings[w2][0]]
    y_vals = [embeddings[w1][1], embeddings[w2][1]]
    z_vals = [embeddings[w1][2], embeddings[w2][2]]
    ax.plot(x_vals, y_vals, z_vals, linestyle='dashed', color='gray')

# Set axis labels
ax.set_xlabel('Άξονας 1')
ax.set_ylabel('Άξονας 2')
ax.set_zlabel('Άξονας 3', labelpad=-1)

# Set title
plt.title("3D Αναπαράσταση Word Embeddings\nCapturing Semantics in Vector Space", fontsize=14, 
          #weight='bold', 
          loc='center')

# Adjust layout for visibility of labels
plt.tight_layout()

# Save high-resolution image
plt.savefig("3D_Word_Embeddings.png", dpi=300)
plt.show()

#Χρονολογική Εξέλιξη των Μοντέλων NLP
import matplotlib.pyplot as plt

# Δεδομένα
phases = [
    {
        "title": "Παραδοσιακοί Αλγόριθμοι (~1995–2012)",
        "bullets": [
            "Χρήση στατιστικών, αραιών αναπαραστάσεων (sparse)",
            "Έλλειψη εννοιολογικής πληροφορίας",
            "Δεν λαμβάνεται υπόψη η σειρά λέξεων"
        ]
    },
    {
        "title": "Νευρωνικά Δίκτυα\nRNN / LSTM / GRU (~2013–2017)",
        "bullets": [
            "Μάθηση από ακολουθιακά δεδομένα",
            "Είσοδοι μέσω embeddings",
            "Προβλήματα: vanishing gradients, long-term dependencies"
        ]
    },
    {
        "title": "Transformers (BERT, GPT, T5)\n2018–σήμερα",
        "bullets": [
            "Παράλληλη επεξεργασία, self-attention contextual embeddings",
            "Κυρίαρχη προσέγγιση σε όλα τα μοντέρνα NLP tasks"
        ]
    }
]

fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
fig.subplots_adjust(top=0.95, bottom=0.00, left=0.05, right=0.95)

# Θέσεις στον άξονα y
y_positions = list(range(len(phases)))[::-1]  # Αντίστροφη σειρά για από πάνω προς τα κάτω

# Ζωγραφίζουμε σημεία & συνδέσεις
for i, (y, phase) in enumerate(zip(y_positions, phases)):
    ax.plot([0], [y], 'o', color='royalblue', markersize=12)
    ax.text(0.1, y - 0.0, phase["title"], fontsize=12, weight='bold', va='bottom')

    for j, bullet in enumerate(phase["bullets"]):
        ax.text(0.2, y - 0.08 - j * 0.15, f"• {bullet}", fontsize=10, va='top')

# Συνδέσεις μεταξύ των φάσεων
for i in range(len(phases) - 1):
    ax.plot([0, 0], [y_positions[i], y_positions[i+1]], color='royalblue', linewidth=2)

# Formatting
ax.set_xlim(-0.2, 2)
ax.set_ylim(-1, len(phases))
ax.axis('off')
#ax.set_title("Χρονολογική Εξέλιξη Εισόδων στα Μοντέλα NLP", fontsize=14, weight='bold', pad=2)
fig.suptitle("Χρονολογική Εξέλιξη Εισόδων στα Μοντέλα NLP", 
             fontsize=14, weight='bold', y=0.79)

#plt.tight_layout()
plt.show()
ax.set_ylim(-1, len(phases))
ax.axis('off')
#ax.set_title("Χρονολογική Εξέλιξη Εισόδων στα Μοντέλα NLP", fontsize=14, weight='bold', pad=2)
fig.suptitle("Χρονολογική Εξέλιξη Εισόδων στα Μοντέλα NLP", 
             fontsize=14, weight='bold', y=0.77)

#plt.tight_layout()
plt.show()

#Διάγραμμα ροής XLM R (feature extraction) → TCN (feature enrichment) → ML #Classifier (prediction)
# Εγκατάσταση graphviz στο Colab
!apt-get install -y graphviz
!pip install graphviz

from graphviz import Digraph

# Δημιουργία διαγράμματος
from graphviz import Digraph

# Δημιουργία διαγράμματος
dot = Digraph(comment='XLM-R -> TCN -> ML Classifier', format='png')
dot.attr(dpi='300')  # Υψηλή ευκρίνεια
dot.attr(rankdir='TB', size='6,8')  # Κάθετη ροή (Top → Bottom)

# Στυλ κόμβων
dot.attr('node', shape='box', style='filled', color='lightblue',
         fontname='Helvetica', fontsize='11', width='2.8', height='0.9')

# Κόμβοι
dot.node('Input', 'Είσοδος Κειμένου\n(Text Input)')
dot.node('Tokenizer', 'Tokenization\n(Προεπεξεργασία Κειμένου)')
dot.node('XLMR', 'XLM-R\nΠροεκπαιδευμένο Transformer\nFeature Extraction')
dot.node('TCN', 'TCN\nTemporal Convolutional Network\nFeature Enrichment')
dot.node('Classifier', 'ML Classifier\n(XGBoost / KNN / LR)\nPrediction')
dot.node('Output', 'Έξοδος\n(Τελική Κατηγορία)')

# Συνδέσεις
dot.edge('Input', 'Tokenizer', label='Κειμενική Είσοδος')
dot.edge('Tokenizer', 'XLMR', label='Ενσωμάτωση tokens')
dot.edge('XLMR', 'TCN', label='Μεταφορά embeddings')
dot.edge('TCN', 'Classifier', label='Εμπλουτισμένα χαρακτηριστικά')
dot.edge('Classifier', 'Output', label='Τελική Πρόβλεψη')

# Αποθήκευση
output_path = '/content/XLMR_TCN_Classifier_Flowchart_Vertical'
dot.render(output_path, format='png', cleanup=True)

print(f"✅ Το διάγραμμα αποθηκεύτηκε στο: {output_path}.png")
print("💡 Συμβουλή: Εισήγαγε το PNG στο Word με 'Εισαγωγή > Εικόνα > Από αρχείο' και ρύθμισε πλάτος ~12–13 cm.")


#Απόδοση Μετρικών σε Datasets 2 κλάσεων (μέσες τιμές)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Δημιουργία των μετρικών για τα τρία datasets με 2 κλάσεις
binary_metrics_data = {
    'Metric': ['Accuracy', 'F1', 'Precision', 'Recall', 'MCC', 'Kappa'],
    'carblacac_2cl': [0.851, 0.851, 0.852, 0.851, 0.703, 0.702],
    'cardiffnlp_2cl': [0.754, 0.754, 0.754, 0.753, 0.507, 0.507],
    'psd_2cl': [0.94, 0.94, 0.94, 0.94, 0.82, 0.82]
}

# Δημιουργία του DataFrame
binary_metrics_df = pd.DataFrame(binary_metrics_data)
binary_metrics_df.set_index('Metric', inplace=True)

# Δημιουργία heatmap
plt.figure(figsize=(10, 6), dpi=300)
sns.heatmap(binary_metrics_df, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, cbar=True)
plt.title("Απόδοση Μετρικών σε Datasets 2 κλάσεων (μέσες τιμές)")
plt.tight_layout()
plt.show()

#Απόδοση Μετρικών σε Datasets τριών κλάσεων (μέσες τιμές)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Δημιουργία των μετρικών για τα trinary datasets
trinary_metrics_data = {
    'Metric': ['Accuracy', 'F1', 'Precision', 'Recall', 'MCC', 'Kappa'],
    'Sp1786_3cl': [0.769, 0.769, 0.772, 0.771, 0.652, 0.652],
    'US_Airlines_3cl': [0.865, 0.862, 0.827, 0.806, 0.737, 0.736],
    'cardiffnlp_3cl': [0.726, 0.726, 0.721, 0.722, 0.565, 0.565],
    'climate_3cl': [0.828, 0.828, 0.834, 0.832, 0.737, 0.737],
    'NusaX_3cl': [0.903, 0.904, 0.895, 0.897, 0.853, 0.853],
    'takala_50_3cl': [0.865, 0.864, 0.857, 0.837, 0.756, 0.754],
    'takala_66_3cl': [0.905, 0.904, 0.914, 0.865, 0.826, 0.825],
    'takala_75_3cl': [0.959, 0.960, 0.945, 0.952, 0.926, 0.925],
    'takala_100_3cl': [0.985, 0.985, 0.980, 0.979, 0.973, 0.973],
    'zeroshot_twitter_3cl': [0.876, 0.877, 0.840, 0.847, 0.766, 0.766],
    'psd_3cl': [0.914, 0.914, 0.914, 0.914, 0.871, 0.871]
}

# Δημιουργία DataFrame
trinary_metrics_df = pd.DataFrame(trinary_metrics_data)
trinary_metrics_df.set_index('Metric', inplace=True)

# Δημιουργία heatmap
plt.figure(figsize=(14, 6), dpi=300)
sns.heatmap(trinary_metrics_df, annot=True, fmt=".3f", cmap="BuPu", linewidths=0.5, cbar=True)
plt.title("Απόδοση Μετρικών σε Datasets 3 κλάσεων μέσες τιμές)")
plt.tight_layout()
plt.show()

#Critical Difference Diagram (Friedman–Nemenyi) για Majority Vote
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

models = [
    "DeBERTa,T5,GPT2", "DeBERTa,Albert,GPT2", "DeBERTa,T5,Albert",
    "DeBERTa,GPT2,Pythia", "DeBERTa,Albert,Pythia", "DeBERTa,T5,Pythia",
    "Bart,Albert,GPT2", "T5,Albert,GPT2", "T5,Albert,Pythia",
    "T5,GPT2,Pythia", "Bart,T5,GPT2", "Bart,Albert,Pythia",
    "Bart,GPT2,Pythia", "Bart,T5,Albert", "Bart,DeBERTa,GPT2",
    "Bart,DeBERTa,T5", "Bart,T5,Pythia", "Bart,DeBERTa,Albert",
    "Bart,DeBERTa,Pythia", "Albert,GPT2,Pythia", "DeBERTa",
    "Bart", "T5", "GPT2", "Albert", "Pythia"
]

ranks = [
    5.333333, 5.666667, 6.750000, 7.250000, 7.750000, 8.083333,
    8.916667, 9.000000, 9.583333, 9.916667, 10.750000, 11.083333,
    11.333333, 11.416667, 12.000000, 13.833333, 14.083333, 14.416667,
    16.916667, 18.916667, 19.416667, 22.166667, 22.416667,
    23.500000, 24.583333, 25.916667
]

CD = 10.802  # Critical Difference από Nemenyi

assert len(models) == len(ranks), "Μη ίδιο μήκος models/ranks"

# Ταξινόμηση για να μπει πρώτος ο ΧΕΙΡΟΤΕΡΟΣ (μεγαλύτερη μέση κατάταξη)
pairs = sorted(zip(models, ranks), key=lambda x: x[1], reverse=True)
labels = [p[0] for p in pairs]
mean_ranks = np.array([p[1] for p in pairs])

half_cd = CD / 2.0
left = mean_ranks - half_cd
right = mean_ranks + half_cd

plt.figure(figsize=(14, 8), dpi=300)

y = np.arange(len(labels))  # τώρα ο χειρότερος πάνω

# Γκρίζες μπάρες: διάστημα CD
for yi, l, r in zip(y, left, right):
    plt.hlines(yi, l, r, color='0.7', linewidth=6, zorder=1)

# Σημεία (μέσες κατατάξεις)
plt.scatter(mean_ranks, y, s=30, color='blue', zorder=2)

# Κάθετη γραμμή στην καλύτερη μέση κατάταξη
best = mean_ranks.min()
plt.axvline(best, linestyle='--', color='green', linewidth=1.5, label='Καλύτερη Μέση Κατάταξη')

plt.yticks(y, labels, fontsize=10.5)
plt.xlabel("Μέση Κατάταξη (Friedman Rank)", fontsize=12)
plt.title("Critical Difference Diagram (Friedman–Nemenyi) για Majority Vote", fontsize=16)

xmin = min(left.min(), mean_ranks.min()) - 0.5
xmax = max(right.max(), mean_ranks.max()) + 0.5
plt.xlim(xmin, xmax)

plt.grid(axis='x', linestyle=':', alpha=0.4)
plt.legend(loc='upper right', frameon=False)
plt.tight_layout()

plt.savefig("cd_majority_vote_reverse.png", dpi=300, bbox_inches='tight')
plt.show()

#Calibration Metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Χρήση παλέτας tab10
palette = sns.color_palette("tab10")

# --- Δεδομένα climate_3cl ---
data_climate = {
    "Model": ["Pythia", "DeBERTa", "GPT-2", "T5"],
    "Average Confidence (Correct)": [0.9256, 0.9340, 0.9129, 0.9372],
    "Average Confidence (Incorrect)": [0.8672, 0.8611, 0.8116, 0.8598],
    "ECE": [0.1807, 0.1157, 0.0894, 0.1092]
}

df_climate = pd.DataFrame(data_climate)

# --- Δεδομένα cardifnlp_2cl ---
data_cardiff = {
    "Model": ["DeBERTa", "Albert", "Pythia", "GPT2"],
    "Average Confidence (Correct)": [0.9719, 0.7252, 0.7066, 0.7941],
    "Average Confidence (Incorrect)": [0.9441, 0.6783, 0.6626, 0.7283],
    "ECE": [0.2138, 0.0325, 0.0420, 0.0937]
}

df_cardiff = pd.DataFrame(data_cardiff)

# Συνάρτηση για δημιουργία γραφήματος
def plot_calibration(df, title, filename):
    fig, ax1 = plt.subplots(figsize=(8,5), dpi=300)

    x = np.arange(len(df["Model"]))
    width = 0.35

    # Ράβδοι για Average Confidence με tab10
    ax1.bar(x - width/2, df["Average Confidence (Correct)"], width, 
            label="Correct", color=palette[0])
    ax1.bar(x + width/2, df["Average Confidence (Incorrect)"], width, 
            label="Incorrect", color=palette[1])

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Average Confidence")
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"])
    ax1.legend(loc="upper left")

    # Δεύτερος άξονας για ECE
    ax2 = ax1.twinx()
    ax2.plot(x, df["ECE"], color=palette[2], marker="o", label="ECE")
    ax2.set_ylabel("ECE")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# Δημιουργία γραφημάτων
plot_calibration(df_climate, "Calibration Metrics - climate_3cl", "calibration_climate_3cl.png")
plot_calibration(df_cardiff, "Calibration Metrics - cardifnlp_2cl", "calibration_cardifnlp_2cl.png")

#---------------------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ορισμός των δεδομένων από κάθε πίνακα
data_23 = {
    "Classifiers": ["MV-“T5,GPT2,Pythia”", "SV-“DeBERTa,T5,GPT2”", "T5"],
    "Accuracy": [0.876, 0.852, 0.828],
    "F1": [0.876, 0.851, 0.828],
    "Precision": [0.877, 0.852, 0.834],
    "Recall": [0.876, 0.851, 0.832],
    "MCC": [0.811, 0.773, 0.737],
    "Kappa": [0.81, 0.773, 0.737]
}

data_24 = {
    "Classifiers": ["MV-Bart,Albert,GPT2", "SV-DeBERTa,T5,Albert", "DeBERTa"],
    "Accuracy": [0.802, 0.741, 0.726],
    "F1": [0.803, 0.741, 0.726],
    "Precision": [0.805, 0.741, 0.721],
    "Recall": [0.802, 0.742, 0.722],
    "MCC": [0.684, 0.588, 0.565],
    "Kappa": [0.683, 0.587, 0.565]
}

data_25 = {
    "Classifiers": ["MV-“Bart,Albert,GPT2”", "SV-“Bart,DeBERTa,T5”", "DeBERTa"],
    "Accuracy": [0.832, 0.777, 0.769],
    "F1": [0.832, 0.776, 0.769],
    "Precision": [0.836, 0.777, 0.772],
    "Recall": [0.832, 0.776, 0.771],
    "MCC": [0.749, 0.664, 0.652],
    "Kappa": [0.747, 0.664, 0.652]
}

data_26 = {
    "Classifiers": ["MV-“T5,GPT2,Pythia”", "SV-“Bart,DeBERTa,T5”", "T5"],
    "Accuracy": [0.917, 0.873, 0.865],
    "F1": [0.916, 0.87, 0.862],
    "Precision": [0.916, 0.869, 0.827],
    "Recall": [0.917, 0.873, 0.806],
    "MCC": [0.833, 0.752, 0.737],
    "Kappa": [0.832, 0.751, 0.736]
}

data_27 = {
    "Classifiers": ["MV-“Bart,T5,GPT2”", "SV-“Bart,DeBERTa,GPT2”", "Bart"],
    "Accuracy": [0.898, 0.886, 0.876],
    "F1": [0.898, 0.885, 0.877],
    "Precision": [0.899, 0.885, 0.840],
    "Recall": [0.898, 0.886, 0.847],
    "MCC": [0.810, 0.781, 0.766],
    "Kappa": [0.809, 0.781, 0.766]
}

data_28 = {
    "Classifiers": ["MV-“Bart,DeBERTa,Albert”", "SV-“Bart,DeBERTa,T5”", "DeBERTa"],
    "Accuracy": [0.933, 0.913, 0.903],
    "F1": [0.933, 0.913, 0.904],
    "Precision": [0.937, 0.913, 0.895],
    "Recall": [0.933, 0.913, 0.897],
    "MCC": [0.901, 0.868, 0.853],
    "Kappa": [0.899, 0.868, 0.853]
}

data_29 = {
    "Classifiers": ["MV-“T5,GPT2,Pythia”", "SV-“Bart,DeBERTa,T5”", "T5"],
    "Accuracy": [0.917, 0.873, 0.865],
    "F1": [0.917, 0.872, 0.864],
    "Precision": [0.918, 0.873, 0.857],
    "Recall": [0.917, 0.873, 0.837],
    "MCC": [0.845, 0.77, 0.756],
    "Kappa": [0.845, 0.768, 0.754]
}

data_30 = {
    "Classifiers": ["MV-“T5,Albert,GPT2”", "SV-“Bart,DeBERTa,Albert”", "DeBERTa"],
    "Accuracy": [0.941, 0.919, 0.905],
    "F1": [0.941, 0.918, 0.904],
    "Precision": [0.941, 0.919, 0.914],
    "Recall": [0.941, 0.919, 0.865],
    "MCC": [0.889, 0.853, 0.826],
    "Kappa": [0.889, 0.852, 0.825]
}

data_31 = {
    "Classifiers": ["MV-“DeBERTa,GPT2,Pythia”", "SV-“Bart,DeBERTa,T5”", "DeBERTa"],
    "Accuracy": [0.968, 0.963, 0.959],
    "F1": [0.968, 0.963, 0.96],
    "Precision": [0.969, 0.963, 0.945],
    "Recall": [0.968, 0.963, 0.952],
    "MCC": [0.942, 0.932, 0.926],
    "Kappa": [0.941, 0.932, 0.925]
}

data_32 = {
    "Classifiers": ["MV-“DeBERTa,Albert,GPT2”", "DeBERTa", "SV-“DeBERTa,T5,GPT2”"],
    "Accuracy": [0.988, 0.985, 0.982],
    "F1": [0.988, 0.985, 0.982],
    "Precision": [0.988, 0.98, 0.982],
    "Recall": [0.988, 0.979, 0.982],
    "MCC": [0.978, 0.973, 0.967],
    "Kappa": [0.978, 0.973, 0.967]
}

data_33 = {
    "Classifiers": ["MV-“DeBERTa,Albert,Pythia”", "DeBERTa", "SV-“DeBERTa,Albert,GPT2”"],
    "Accuracy": [0.839, 0.754, 0.754],
    "F1": [0.837, 0.754, 0.727],
    "Precision": [0.845, 0.754, 0.761],
    "Recall": [0.839, 0.753, 0.696],
    "MCC": [0.68, 0.507, 0.505],
    "Kappa": [0.672, 0.507, 0.503]
}

data_34 = {
    "Classifiers": ["MV-“DeBERTa,Albert,Pythia”", "SV-“Bart,DeBERTa,T5”", "DeBERTa"],
    "Accuracy": [0.905, 0.857, 0.851],
    "F1": [0.905, 0.857, 0.851],
    "Precision": [0.907, 0.86, 0.852],
    "Recall": [0.905, 0.853, 0.851],
    "MCC": [0.811, 0.715, 0.703],
    "Kappa": [0.81, 0.715, 0.702]
}

# Δημιουργία λίστας με όλα τα datasets
all_data = [
    (23, data_23, "climate_3cl"),
    (24, data_24, "cardiffnlp_3cl"),
    (25, data_25, "Sp1786_3cl"),
    (26, data_26, "US_Airlines_3cl"),
    (27, data_27, "0shot_twitter_3cl"),
    (28, data_28, "NusaX_3cl"),
    (29, data_29, "takala_50_3cl"),
    (30, data_30, "takala_66_3cl"),
    (31, data_31, "takala_75_3cl"),
    (32, data_32, "takala_100_3cl"),
    (33, data_33, "cardiffnlp_2cl"),
    (34, data_34, "carblacac_2cl")
]

# Ορισμός παλέτας χρωμάτων για 3 classifiers
color_palette = sns.color_palette("tab10", 3)

# Δημιουργία γραφημάτων για κάθε πίνακα με ανάποδη διάταξη
for table_num, data, dataset_name in all_data:
    df = pd.DataFrame(data)

    # Μετατροπή του DataFrame σε μακρύ format
    df_melted = df.melt(id_vars='Classifiers',
                        value_vars=['Accuracy', 'F1', 'Precision', 'Recall', 'MCC', 'Kappa'],
                        var_name='Metric',
                        value_name='Score')

    # Δημιουργία του subplot με μεγαλύτερο μέγεθος για καλύτερη ευκρίνεια
    plt.figure(figsize=(14, 8))

    # Δημιουργία bar plot με μετρικές στον άξονα x και classifiers ως hue
    ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Classifiers',
                     palette=color_palette, edgecolor='black', width=0.8)

    # Προσθήκη τιμών πάνω από τα bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)

    # Ρύθμιση του τίτλου και labels
    plt.title(f'Πίνακας {table_num}: Σύνολα έναντι του νικητή των μετασχηματιστών για το {dataset_name}',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Μετρικές', fontsize=14)
    plt.ylabel('Score', fontsize=14)

    # Ρύθμιση των ticks στον άξονα y
    plt.ylim(0, 1.05)

    # Ρύθμιση θέσης legend
    plt.legend(title='Classifiers', title_fontsize=12, fontsize=11,
               loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=3)

    # Ρύθμιση των labels στον άξονα x
    plt.xticks(fontsize=12, rotation=0)

    # Ρύθμιση γραμμών πλέγματος
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Ρύθμιση layout για να χωράει το legend
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Αποθήκευση της εικόνας σε 300 dpi
    plt.savefig(f'Πίνακας_{table_num}_{dataset_name}_ανάποδα.png', dpi=300, bbox_inches='tight')

    # Εμφάνιση του γραφήματος
    plt.show()

    print(f"Δημιουργήθηκε το γράφημα για τον Πίνακα {table_num}: {dataset_name}")

# Εναλλακτική έκδοση: Όλα τα γραφήματα σε ένα μόνο figure με subplots
print("\n--- Δημιουργία όλων των γραφημάτων σε ένα μόνο figure ---")

# Δημιουργία figure με πολλαπλά subplots
fig, axes = plt.subplots(4, 3, figsize=(24, 28))
fig.suptitle('Σύνολα έναντι του νικητή των μετασχηματιστών για όλα τα datasets\n(Μετρικές στον άξονα Χ, Classifiers ως διαφορετικές μπάρες)',
             fontsize=22, fontweight='bold', y=1.02)

axes = axes.flatten()

for idx, (table_num, data, dataset_name) in enumerate(all_data):
    if idx >= len(axes):
        break

    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Classifiers',
                        value_vars=['Accuracy', 'F1', 'Precision', 'Recall', 'MCC', 'Kappa'],
                        var_name='Metric',
                        value_name='Score')

    ax = axes[idx]
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Classifiers',
                palette=color_palette, edgecolor='black', ax=ax, width=0.8)

    ax.set_title(f'Πίνακας {table_num}: {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Μετρικές', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)

    # Προσθήκη τιμών (μόνο για το πρώτο bar κάθε ομάδας για ευκρίνεια)
    if idx == 0:
        for container in ax.containers[:1]:  # Μόνο το πρώτο classifier για να μην γίνει πολύ γεμάτο
            ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)

    # Ρύθμιση legend
    if idx == 0:  # Προσθήκη legend μόνο στο πρώτο γράφημα
        ax.legend(title='Classifiers', title_fontsize=11, fontsize=10,
                  loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    else:
        ax.get_legend().remove()

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)

# Ρύθμιση layout
plt.tight_layout()

# Αποθήκευση του συνολικού figure
plt.savefig('Όλοι_οι_πίνακες_ανάποδα.png', dpi=300, bbox_inches='tight')
plt.show()

print("Δημιουργήθηκαν όλα τα γραφήματα!")

# Δημιουργία πιο συμπαγών γραφημάτων με καλύτερη οργάνωση - ΒΕΛΤΙΩΜΕΝΗ ΕΚΔΟΣΗ
for table_num, data, dataset_name in all_data:
    df = pd.DataFrame(data)

    # Μετατροπή του DataFrame σε μακρύ format
    df_melted = df.melt(id_vars='Classifiers',
                        value_vars=['Accuracy', 'F1', 'Precision', 'Recall', 'MCC', 'Kappa'],
                        var_name='Metric',
                        value_name='Score')

    # Δημιουργία του subplot - ΠΙΟ ΚΟΝΤΟ ΓΙΑ WORD
    fig, ax = plt.subplots(figsize=(10, 5))

    # Δημιουργία grouped bar plot
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'MCC', 'Kappa']
    classifiers = df['Classifiers'].tolist()

    # Δημιουργία θέσεων για κάθε ομάδα μπάρας
    x = np.arange(len(metrics))
    width = 0.25  # Πλάτος κάθε μπάρας

    # Δημιουργία μπαρών για κάθε classifier
    for i, classifier in enumerate(classifiers):
        # Εξαγωγή τιμών για τον τρέχοντα classifier
        values = [df[df['Classifiers'] == classifier][metric].values[0] for metric in metrics]
        # Θέση για κάθε ομάδα μπαρών
        positions = x + (i - 1) * width
        bars = ax.bar(positions, values, width, label=classifier,
                      color=color_palette[i], edgecolor='black', alpha=0.8)

        # Προσθήκη τιμών πάνω από τα bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    # Ρύθμιση του τίτλου - 1 ΕΚΑΤΟΣΤΟ ΑΠΟ ΤΟ ΓΡΑΦΗΜΑ
    # Μετατροπή εκατοστών σε inches (1 εκατοστό = 0.3937 inches)
    cm_to_inch = 0.3937
    title_distance_cm = 1  # 1 εκατοστό
    total_height = 5  # ύψος γραφήματος σε inches

    # Υπολογισμός pad σε inches
    title_pad_inches = title_distance_cm * cm_to_inch
    # Μετατροπή σε points (1 inch = 72 points)
    title_pad_points = title_pad_inches * 72

    ax.set_title(f'Πίνακας {table_num}: Σύνολα έναντι του νικητή για το {dataset_name}',
                 fontsize=14, fontweight='bold', pad=title_pad_points)

    # Ρύθμιση labels
    ax.set_xlabel('Μετρικές', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)

    # ΑΦΑΙΡΕΣΗ ΠΛΑΙΣΙΟΥ - ΜΟΝΟ ΑΞΟΝΕΣ Χ,Υ
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # Ρύθμιση legend - 1.5 ΕΚΑΤΟΣΤΑ ΑΠΟΣΤΑΣΗ ΑΠΟ ΤΟ ΓΡΑΦΗΜΑ (1 + 0.5)
    # Υπολογισμός του bbox_to_anchor για legend
    legend_distance_cm = 1.5  # 1.5 εκατοστά (1 + 0.5 επιπλέον)
    legend_y_position = 1 + (legend_distance_cm * cm_to_inch / total_height)

    # ΜΕΓΑΛΥΤΕΡΗ ΓΡΑΜΜΑΤΟΣΕΙΡΑ ΓΙΑ ΤΑ CLASSIFIERS
    legend = ax.legend(title='Classifiers', title_fontsize=12, fontsize=12,  # fontsize=12
              loc='upper center', bbox_to_anchor=(0.5, legend_y_position),  # 1.5 εκατοστά
              fancybox=False, shadow=False, ncol=3, frameon=False)

    # Διαφάνεια στο title του legend
    legend.get_title().set_backgroundcolor((1, 1, 1, 0))  # Διαφανές white background
    legend.get_title().set_color('black')

    # Ρύθμιση γραμμών πλέγματος - ΜΟΝΟ ΟΡΙΖΟΝΤΙΕΣ
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.grid(axis='x', visible=False)  # Απενεργοποίηση κατακόρυφων γραμμών

    # Ρύθμιση layout - Αφήνουμε περισσότερο χώρο πάνω για τίτλο και legend
    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.1, right=0.95)  # top=0.90 για περισσότερο χώρο

    # Αποθήκευση της εικόνας
    plt.savefig(f'Πίνακας_{table_num}_{dataset_name}_grouped_clean.png', dpi=300, bbox_inches='tight',
                transparent=True)

    # Εμφάνιση του γραφήματος
    plt.show()

print("Δημιουργήθηκαν όλα τα βελτιωμένα grouped γραφήματα με legend 1.5 εκατοστά από το γράφημα!")
