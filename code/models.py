import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, TopKCategoricalAccuracy, Precision, \
    Recall, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from keras import backend as K
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

# import external modules
import utils


def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



class DetectionCNN:
    def __init__(self, args):
        self.save_dir = args['save_dir']
        self.cnn = args['cnn']
        self.height = args['height']
        self.width = args['width']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.loss = args['loss']
        self.optimizer = Adam(learning_rate=args['learning_rate'])
        self.eval_metrics = [BinaryAccuracy(name="accuracy"), Precision(name="precision"),
                             Recall(name="recall"), TruePositives(name='tp'), 
                             FalsePositives(name='fp'), TrueNegatives(name='tn'),
                             FalseNegatives(name='fn')]
        self.vAcc, self.vLoss, self.vTP, self.vFP, self.vTN, \
            self.vFN, self.vPrec, self.vRec = ([] for _ in range(8))

        self.model = self.__build()


    def get_cnn(self):
        if self.cnn == "ResNet":
            return ResNet50(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
        elif self.cnn == "Mobile":
            return MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
        elif self.cnn == "VGG":
            return VGG16(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
        else:
            sys.exit("Incert valid cnn model. Options: Mobile, ResNet or VGG (case sensitive)")


    def __build(self):
        base_cnn = self.get_cnn()
        x = GlobalAveragePooling2D()(base_cnn.output)
        z = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=base_cnn.input, outputs=z)
        model.compile(optimizer = self.optimizer, 
                      loss = self.loss, 
                      metrics = self.eval_metrics)
        return model


    def train(self, train_dataset, val_dataset, fold_var):
        dt = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=False),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(os.path.join(self.save_dir, utils.get_model_name(fold_var)), verbose=1, save_best_only=True),
            TensorBoard(log_dir=os.path.join(self.save_dir, "logs", dt))
        ]

        history = self.model.fit(
            x = train_dataset,
            steps_per_epoch = len(train_dataset),
            validation_data = val_dataset,
            validation_steps = len(val_dataset), 
            epochs = self.epochs,
            callbacks = callbacks,
            verbose = 1, 
            class_weight = {0:1.6, 1:1},
            shuffle = True
        )
        return history


    def get_eval(self, val_dataset, fold_var):
        self.model.load_weights(os.path.join(self.save_dir, utils.get_model_name(fold_var)))
        results = self.model.evaluate(
            val_dataset, 
            steps = len(val_dataset)
        )
        results = dict(zip(self.model.metrics_names, results))

        self.vAcc.append(results['accuracy'])
        self.vLoss.append(results['loss'])
        self.vTP.append(results["tp"])
        self.vFP.append(results["fp"])
        self.vTN.append(results["tn"])
        self.vFN.append(results["fn"])
        self.vPrec.append(results["precision"])
        self.vRec.append(results["recall"])
        

    def get_preds(self, val_dataset, val_labels, fold_var):
        predictions = self.model.predict(val_dataset)
        predictions = tf.where(predictions < 0.5, 0, 1)

        cm = confusion_matrix(val_labels, predictions, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        cm_display = disp.plot()
        plt.savefig(f'{self.save_dir}/confusion_matrix_{fold_var}.jpg', bbox_inches='tight')


    def save_metrics(self):
        csv_dir = os.path.join(self.save_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        metrics = {'test_accuracy':self.vAcc, 'test_loss':self.vLoss, 'test_tp':self.vTP, 
                   'test_fp':self.vFP, 'test_tn':self.vTN, 'test_fn':self.vFN, 
                   'test_precision':self.vPrec, 'test_recall':self.vRec}
        
        for m in metrics:
            with open(os.path.join(csv_dir, m+'.pkl'), 'wb') as handle:
                pickle.dump(metrics[m], handle, protocol=pickle.HIGHEST_PROTOCOL)


    def __reset(self):
        self.model = self.__build()




class CategorizationCNN:
    def __init__(self, args):
        self.save_dir = args['save_dir']
        self.cnn = args['cnn']
        self.height = args['height']
        self.width = args['width']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.loss = args['loss']
        self.eye = args['eye_only']
        self.dropout = args['dropout']
        self.drop_rate = args['drop_rate']
        self.optimizer = Adam(learning_rate=args['learning_rate'])
        self.eval_metrics = [CategoricalAccuracy(name="accuracy"), TopKCategoricalAccuracy(k=2, name="top2_accuracy"), 
                             Precision(name="precision"), Recall(name="recall"), 
                             TruePositives(name='tp'), FalsePositives(name='fp'),
                             TrueNegatives(name='tn'), FalseNegatives(name='fn')]
        self.vAcc, self.vTop2Acc, self.vLoss, self.vTP, self.vFP, self.vTN, \
            self.vFN, self.vPrec, self.vRec = ([] for _ in range(9))

        self.ft = args['finetune']
        if not self.ft:
            self.model = self.__build()
        else:
            self.init_epochs = args['initial_epochs']
            self.ft_epochs = args['finetune_epochs']
            self.ft_at = args['finetune_at']
            assert isinstance(self.ft_at, int)
            self.base_cnn = self.get_cnn()
            self.model = self.__buildftStage1()


    def get_cnn(self):
        if self.cnn == "ResNet":
            return ResNet50(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
        elif self.cnn == "Mobile":
            return MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
        elif self.cnn == "VGG":
            return VGG16(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
        else:
            sys.exit("Incert valid cnn model. Options: Mobile, ResNet or VGG (case sensitive)")


    def __build(self):
        base_cnn = self.get_cnn()
        x = GlobalAveragePooling2D()(base_cnn.output)
        x = Dropout(self.drop_rate)(x) if self.dropout else x

        classes = 6 if not self.eye else 5
        z = Dense(classes, activation="softmax")(x)

        model = Model(inputs=base_cnn.input, outputs=z)
        model.compile(optimizer = self.optimizer, 
                      loss = self.loss,
                      # weight the loss contributions of different model outputs as 1:1
                      loss_weights = np.ones(len(self.loss)).tolist() if len(self.loss) > 1 else None,
                      metrics = self.eval_metrics)
        return model


    def train(self, train_dataset, val_dataset, class_weights, fold_var):
        dt = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=False),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(os.path.join(self.save_dir, utils.get_model_name(fold_var)), verbose=1, save_best_only=True),
            #TensorBoard(log_dir=os.path.join(self.save_dir, "logs", dt))
        ]

        history = self.model.fit(
            x = train_dataset,
            steps_per_epoch = len(train_dataset),
            validation_data = val_dataset,
            validation_steps = len(val_dataset), 
            epochs = self.epochs,
            callbacks = callbacks,
            verbose = 1, 
            class_weight = class_weights,
            shuffle = True
        )
        return history


    #########################################
    ######         FINE-TUNING         ######
    #########################################
    def __buildftStage1(self):
        # freeze base model
        self.base_cnn.trainable = False

        inputs = Input(shape=(self.width, self.height, 3))
        x = self.base_cnn(inputs, training=False)
        x = GlobalAveragePooling2D()(x.output)
        x = Dropout(self.drop_rate)(x) if self.dropout else x

        classes = 6 if not self.eye else 5
        z = Dense(classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=z)
        model.compile(optimizer = self.optimizer,
                      loss = self.loss,
                      # weight the loss contributions of different model outputs as 1:1
                      loss_weights = np.ones(len(self.loss)).tolist() if len(self.loss) > 1 else None,
                      metrics = self.eval_metrics)
        return model

    
    def __buildftStage2(self):
        # unfreeze base model
        self.base_cnn.trainable = True

        if self.ft_at > 0:
            # input is given as layer number
            # fine-tune from this layer onwards, freezing all layers before
            for layer in self.base_cnn.layers[:self.ft_at]:
                layer.trainable = False
        else:
            # input is given as number of final layers to finetune
            nb_layers2ft = abs(self.ft_at)
            total2freeze = len(self.base_cnn.layers) - nb_layers2ft
            for layer in self.base_cnn.layers[:total2freeze]:
                layer.trainable = False

        self.model.compile(optimizer = Adam(learning_rate=1e-5), 
                           loss = self.loss,
                           # weight the loss contributions of different model outputs as 1:1
                           loss_weights = np.ones(len(self.loss)).tolist() if len(self.loss) > 1 else None,
                           metrics = self.eval_metrics)


    def trainftStage1(self, train_dataset, val_dataset, class_weights, fold_var):
        # train 1st stage
        dt = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        callbacks = [
            ModelCheckpoint(os.path.join(self.save_dir, f"best_model_frozen_{fold_var}.h5"), verbose=1, save_best_only=True),
            #TensorBoard(log_dir=os.path.join(self.save_dir, "logs", dt))
        ]
        
        history = self.model.fit(
            train_dataset,
            steps_per_epoch = len(train_dataset),
            validation_data = val_dataset,
            validation_steps = len(val_dataset),
            epochs = self.init_epochs,
            callbacks = callbacks,
            class_weight = class_weights,
            shuffle = True
        )
        return history

    
    def trainftStage2(self, train_dataset, val_dataset, class_weights, prev_hist, fold_var):
        # train 2nd stage
        dt = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        callbacks = [
            ModelCheckpoint(os.path.join(self.save_dir, f"best_model_finetuned_{fold_var}.h5"), verbose=1, save_best_only=True),
            #TensorBoard(log_dir=os.path.join(self.save_dir, "logs", dt))
        ]
        
        total_epochs = self.init_epochs + self.ft_epochs

        history = self.model.fit(
            train_dataset,
            steps_per_epoch = len(train_dataset),
            validation_data = val_dataset,
            validation_steps = len(val_dataset),
            epochs = total_epochs,
            initial_epoch = prev_hist.epoch[-1],
            callbacks = callbacks,
            class_weight = class_weights,
            shuffle = True
        )
        return history
    #########################################


    def get_eval(self, val_dataset, fold_var):
        name = utils.get_model_name(fold_var) if not self.ft else f"best_model_finetuned_{fold_var}.h5"
        self.model.load_weights(os.path.join(self.save_dir, name))
        results = self.model.evaluate(
            val_dataset, 
            steps = len(val_dataset)
        )
        results = dict(zip(self.model.metrics_names, results))

        self.vAcc.append(results['accuracy'])
        self.vTop2Acc.append(results['top2_accuracy'])
        self.vLoss.append(results['loss'])
        self.vTP.append(results["tp"])
        self.vFP.append(results["fp"])
        self.vTN.append(results["tn"])
        self.vFN.append(results["fn"])
        self.vPrec.append(results["precision"])
        self.vRec.append(results["recall"])
        

    def get_preds(self, val_dataset, val_labels, fold_var):
        predictions = self.model.predict(val_dataset)
        predictions_non_category = [np.argmax(t) for t in predictions]
        val_labels_non_category = [np.argmax(t) for t in val_labels]

        labels = [0,1,2,3,4,5] if not self.eye else [0,1,2,3,4]
        display_labels = labels if not self.eye else [x+1 for x in labels]
        cm = confusion_matrix(val_labels_non_category, predictions_non_category, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        cm_display = disp.plot()        
        plt.savefig(f'{self.save_dir}/confusion_matrix_{fold_var}.jpg', bbox_inches='tight')
        

    def save_metrics(self):
        csv_dir = os.path.join(self.save_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        metrics = {'test_accuracy':self.vAcc, 'test_top2_accuracy':self.vTop2Acc, 'test_loss':self.vLoss, 
                   'test_tp':self.vTP, 'test_fp':self.vFP, 'test_tn':self.vTN,
                   'test_fn':self.vFN, 'test_precision':self.vPrec, 'test_recall':self.vRec}
        
        for key, val in metrics.items():
            with open(os.path.join(csv_dir, key+'.pkl'), 'wb') as handle:
                pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def __reset(self):
        if not self.ft:
            self.model = self.__build()
        else:
            self.base_cnn = self.get_cnn()
            self.model = self.__buildftStage1()