import pandas as pd
from transformers import TFBertModel
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf

class BertClassifier(tf.keras.Model):
    def __init__(self, bert: TFBertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = Dense(num_classes, activation='sigmoid')

    @tf.function
    def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        cls_output = outputs[1]
        cls_output = self.classifier(cls_output)

        return cls_output

label_cols = ['toxic']

df_test = pd.read_csv("test.csv")
model = BertClassifier(TFBertModel.from_pretrained('BERT-base_best.pt'), len(label_cols))

test_input_ids = tokenize_sentences(df_test['comment_text'], tokenizer, MAX_LEN)
test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
test_attention_masks = create_attention_masks(test_input_ids)

TEST_BATCH_SIZE = 32
test_steps = len(df_test) // TEST_BATCH_SIZE

test_dataset = create_dataset((test_input_ids, test_attention_masks), batch_size=TEST_BATCH_SIZE, train=False, epochs=1)

df_submission = pd.read_csv(subm_path, index_col='id')
subm_path = 'data/sample_submission.csv'
for i, (token_ids, masks) in enumerate(tqdm(test_dataset, total=test_steps)):
    sample_ids = df_test.iloc[i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE]['id']
    predictions = model(token_ids, attention_mask=masks).numpy()

    df_submission.loc[sample_ids, label_cols] = predictions