MODEL = "distilbert-base-uncased"
TRAINED_MODEL="sentiment_classification"
class TrainConfig:
    EPOCHS = 10 
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5 
    PER_DEVICE_TRAIN_BATCH_SIZE=16
    PER_DEVICE_EVAL_BATCH_SIZE=16
    NUM_TRAIN_EPOCHS=5
    WEIGHT_DECAY=0.01


