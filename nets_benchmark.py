import os
import pandas as pd
from conv_models.convolutional_net import ConvolutionalNet

training_path = "I:\\Dataset\\ROI DAMAGED\\ROI Classes"
test_path = "I:\\Dataset\\ROI DAMAGED\\ROI Classes\\test"

models_path = "models_weights/"
EPOCHS = 10
LEARNING_RATE = 0.000001  # 1 E-6
BATCH_SIZE = 16

if __name__ == "__main__":
    descriptions = []
    models = [#ConvolutionalNet(training_path, 'ResNet50', 'gpu'),
              #ConvolutionalNet(training_path, 'ResNext50', 'gpu'),
              ConvolutionalNet(training_path, 'RexNet', 'cpu')]

    for model in models:
        model_description = {}
        for epoch in range(EPOCHS):
            time_tot, train_map, train_loss, _ = model.train(epochs=1, lr=LEARNING_RATE,
                                                             batch_size=BATCH_SIZE)
            test_time, test_metric = model.test(batch_size=BATCH_SIZE)

            valid_loss, metric_score, time_tot_test, _ = model.validate_path('test_images')

            # Build Models Description
            model_description[epoch] = {'name': model.name, 'training_time': time_tot, 'training_loss': train_loss,
                                 'test_time': test_time, 'f1': test_metric['f1_score'], 'acc': test_metric['accuracy'], # 'epoch_description': train_map,
                                'precision': test_metric['precision'], 'recall': test_metric['recall'],
                                 'f1_test_yolo' : metric_score['f1_score'], 'acc_test_yolo' : metric_score['accuracy'],
                                'prec_test_yolo' : metric_score['precision'], 'rec_test_yolo' : metric_score['recall']}

            # Save Models on a Path
            if not os.path.exists(models_path):
                os.makedirs(models_path)
        model.save_model(models_path)
        df = pd.DataFrame(model_description).transpose()
        df.to_csv(models_path + '/REXNET_BENCH.csv')


'''
    # Concat The Benchmark Dataframe and save it
    benchmarks = None
    if os.path.exists(models_path + '/models_benchmark.csv'):
        benchmarks = pd.read_csv(models_path + '/models_benchmark.csv')

    df = pd.DataFrame(descriptions)
    df['EPOCHS'] = EPOCHS
    df['LEARNING_RATE'] = LEARNING_RATE
    df['BATCH_SIZE'] = BATCH_SIZE

    if benchmarks is not None:
        df = pd.concat([df, benchmarks], )

    df.to_csv(models_path + '/models_benchmark.csv')
'''
