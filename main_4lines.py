import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from unet_210204 import *
from f_measure import f_measure, f_measure_loss
from images_loader import *
from option_parser import get_option
import argparse
import time
import sys
from more_itertools import chunked
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

# for parallel GPUs
import tensorflow as tf
#from keras.utils.multi_gpu_utils import multi_gpu_model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', type = int, default = 300) #If epochs = -1, then early stopping.
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--whole_prediction', action='store_true')
parser.add_argument('--set_fold', type = int, default = -1) #IF set_fold = -1, calc all folds. 
parser.add_argument('--add_drop_layer', action='store_true') #If set --add_drop_layer, it becoms True.
parser.add_argument('--original_color', action='store_true')
parser.add_argument('--rotation', action='store_true')
parser.add_argument('--dropout_ratio', type = float, default = 0)
parser.add_argument('--optimizer', default = 'Adam')
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--GPU_ID', default = '0')
parser.add_argument('--batch_size', type = int, default = 4)
parser.add_argument('--model', default = 'Unet')
#parser.add_argument('--modified_data', action = 'store_true')
parser.add_argument('--weight', default = 'NW')
parser.add_argument('--reduction', type = int, default = 1) #学習データ数制御用. 2だと1/2, 4だと1/4.
parser.add_argument('--train_dir') #Directory of trainig dataset. E.g., L1, L12,
parser.add_argument('--test_dir') #Directory of test dataset. E.g., L1, L12
parser.add_argument('--load_model') #Directory of trained network
parser.add_argument('--add_drop_layer_en', action='store_true') 
parser.add_argument('--layer_num', type = int, default = 1)
parser.add_argument('--BN_position', type = int, default = 1)
parser.add_argument('--test_dif_line', action = 'store_true')
parser.add_argument('--nb_class', type = int, default = 5)
parser.add_argument('--load_output_dir')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ID #"0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5"
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)


INPUT_IMAGE_SIZE = 512
BATCH_SIZE = args.batch_size
if args.epochs > 0:
    EPOCHS = args.epochs  #original = 300
else:
    EPOCHS = 300
EPOCH_PATIENCE = 50
START_NUM = 1
TEST_DELTA = 40
TEST_SPLIT_NUM = 5
nb_class = args.nb_class
END_NUM = TEST_DELTA * TEST_SPLIT_NUM
DO_TRAIN = args.train
DO_TEST = args.test
DO_WHOLE_PREDICTION = args.whole_prediction
DO_DROPOUT = args.add_drop_layer
DROPOUT_RATIO = args.dropout_ratio
OPTIMIZER = args.optimizer
LR = args.lr
SET_FOLD = args.set_fold
MODEL = args.model
ORIGINAL_COLOR = args.original_color
ROTATION = args.rotation
#MODIFIED_DATA = args.modified_data
WEIGHT = args.weight
REDUCTION = args.reduction
TRAIN_DIR = args.train_dir
TEST_DIR = args.test_dir
LOAD_MODEL_DIR = args.load_model
DO_DROPOUT_EN = args.add_drop_layer_en
LAYER_NUM = args.layer_num
BN_POSITION = args.BN_position
TEST_DIF_LINE = args.test_dif_line
LOAD_OUTPUT_DIR = args.load_output_dir

if not DO_WHOLE_PREDICTION and not TEST_DIF_LINE:
    PARAMETER_DISC = 'TRAIN'+TRAIN_DIR+'_TEST'+TEST_DIR+('' if REDUCTION == 1 else '_RED'+str(REDUCTION))+'_'+ ('' if nb_class == 5 else 'class'+str(nb_class)+'_') +MODEL+'_LNUM'+str(LAYER_NUM)+'_BNPos'+str(BN_POSITION)+('_ROT' if ROTATION else '_noRotation')+ '_epoch'+str(args.epochs)+('' if BATCH_SIZE == 4 else ('_BS'+str(BATCH_SIZE)))+'_DO'+str(DO_DROPOUT)+'_DOR'+str(DROPOUT_RATIO)+'_DOEN'+str(DO_DROPOUT_EN)+'_'+ WEIGHT + ('' if OPTIMIZER == 'Adam' else ('_'+OPTIMIZER))+'_LR'+str(LR)


    DIR_INPUTS = os.path.join('./',  '1207_S1_'+TRAIN_DIR+'_labeloriginal_rotation' )
    DIR_TEACHERS = os.path.join('./', '1207_S1_'+TRAIN_DIR+'_converted_label_rotation')
    DIR_TESTS = os.path.join('./', '1207_S1_'+TEST_DIR+'_labeloriginal_rotation')
    DIR_TEST_LABEL = os.path.join('./', '1207_S1_'+TEST_DIR+'_converted_label_rotation')

    DIR_MODEL = os.path.join('./', 'model'+PARAMETER_DISC)
    DIR_OUTPUTS = os.path.join('./', 'result'+PARAMETER_DISC)

    if not(os.path.exists(DIR_MODEL)):
       os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
       os.mkdir(DIR_OUTPUTS)


def vec2prob(vec):
    return vec/np.sum(vec)

def train(TEST_LIST, TRAIN_LIST, TEST_NUM):
    print('DIR_INPUTS', DIR_INPUTS)
    (train_files, inputs) = load_images_train(DIR_INPUTS, INPUT_IMAGE_SIZE, END_NUM, TRAIN_LIST, TEST_LIST, ORIGINAL_COLOR, ROTATION)
    print('train_img_files:', train_files)
    (train_files, teachers) = load_images_train_l(DIR_TEACHERS, INPUT_IMAGE_SIZE, END_NUM, TRAIN_LIST, TEST_LIST, ORIGINAL_COLOR, ROTATION, nb_class)
    print('train_label_files:', train_files)
    if args.epochs < 0:
        train_validation_ratio = 0.8
        random_list = np.random.permutation(range(len(inputs)))
        train_index_list = random_list[:int(len(inputs)*train_validation_ratio)]
        validation_index_list = random_list[int(len(inputs)*train_validation_ratio):]
        print('train_index_list', train_index_list, args.epochs )
        print('validation_index_list', validation_index_list)
        
        class_freq = np.array([np.sum(teachers[train_index_list].argmax(axis=3) == i) for i in range(nb_class)])
    else:
        class_freq = np.array([np.sum(teachers.argmax(axis=3) == i) for i in range(nb_class)])
    #with tf.device("/cpu:0"): 
    #    network = UNet(INPUT_IMAGE_SIZE)
    #    model = network.model()
    
    class_weights = np.max(class_freq) /(class_freq+1)
    class_weights_sqrt = np.sqrt(class_weights)    
    class_weights_inv5 = (class_weights - 1)/5+1
    print('np.log(class_freq)', np.log(class_weights)+1)
    class_weights_log = np.log(class_weights)+1
    #class_weights = vec2prob(class_weights)
    #class_weights_sqrt = vec2prob(class_weights_sqrt)
    #class_weights_inv5 = vec2prob(class_weights_inv5)
    #class_weights_log = vec2prob(class_weights_log)
    print('class_freq', class_freq, 'class_weights', class_weights, 'class_weights sqrt', class_weights_sqrt,'log', class_weights_log, 'INV5', class_weights_inv5)
    
    def crossentropy(y_true, y_pred):
        return K.mean(-K.sum(y_true*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])
    def weighted_crossentropy(y_true, y_pred):
        return K.mean(-K.sum((y_true*class_weights)*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])
    def weighted_crossentropy_sqrt(y_true, y_pred):
        return K.mean(-K.sum((y_true*class_weights_sqrt)*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])
    def weighted_crossentropy_inv5(y_true, y_pred):
        return K.mean(-K.sum((y_true*class_weights_inv5)*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])
    def weighted_crossentropy_log(y_true, y_pred):
        return K.mean(-K.sum((y_true*class_weights_log)*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])

    if MODEL == 'Unet':
        network = UNet(INPUT_IMAGE_SIZE, add_drop_layer=DO_DROPOUT, dropout_ratio = DROPOUT_RATIO, add_drop_layer_encoder=DO_DROPOUT_EN, layer_num = LAYER_NUM, BN=BN_POSITION, nb_class = nb_class)
        model = network.model()
    elif MODEL == 'SegNet':
        model = segnet(INPUT_IMAGE_SIZE, 2)
    #gpu_count = 2
    #pararrel_model = multi_gpu_model(model, gpus=gpu_count)
    
    #pararrel_model.compile(optimizer='adam', loss=f_measure_loss)    
    #history = pararrel_model.fit(inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1) 
    #model.compile(optimizer='adam', loss=f_measure_loss, lr = LR)

    #print('input', inputs[0])
    #print('label', teachers[0])

    if OPTIMIZER == 'Adam':
        if WEIGHT == 'NW':
            model.compile(optimizer=Adam(lr = LR), loss=crossentropy)
        elif WEIGHT == 'INV':
            model.compile(optimizer=Adam(lr = LR), loss=weighted_crossentropy)
        elif WEIGHT == 'SQRT':
            model.compile(optimizer=Adam(lr = LR), loss=weighted_crossentropy_sqrt)
        elif WEIGHT == 'INV5':
            model.compile(optimizer=Adam(lr = LR), loss=weighted_crossentropy_inv5)
        elif WEIGHT == 'LOG':
            model.compile(optimizer=Adam(lr = LR), loss=weighted_crossentropy_log)
        #else:
        #    model.compile(optimizer=Adam(lr = LR), loss=f_measure_loss)
    elif OPTIMIZER == 'AdaBound':
        model.compile(optimizer=AdaBound(lr=LR, final_lr=0.1), loss=f_measure_loss)
    elif OPTIMIZER == 'AMSBound':
        model.compile(optimizer=AdaBound(lr=LR, final_lr=0.1, amsbound = True), loss=f_measure_loss)
    #model.compile(optimizer=Adam(lr = LR), loss=f_measure_loss) 
    if args.epochs > 0:
        history = model.fit(inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1) 
        model.save(os.path.join(DIR_MODEL, File_MODEL))
    else:
        checkpointer = ModelCheckpoint(filepath=os.path.join(DIR_MODEL, File_MODEL), verbose=1, save_best_only=True)
        history = model.fit(inputs[train_index_list], teachers[train_index_list], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                        validation_data = (inputs[validation_index_list], teachers[validation_index_list]), callbacks=[EarlyStopping(monitor='val_loss',min_delta=0,patience=EPOCH_PATIENCE), checkpointer])
    plotLearningCurve(history, TEST_NUM)



def plotLearningCurve(history, TEST_NUM):
    pyplot.figure()
    x = range(len(history.history['loss']))
    pyplot.plot(x, history.history['loss'], label="loss")
    if args.epochs < 0:
        pyplot.plot(x, history.history['val_loss'], label="val_loss")
    pyplot.title("loss")
    pyplot.legend()
    pyplot.savefig(DIR_OUTPUTS+'/LC_epoch'+str(EPOCHS)+'_DO'+str(DO_DROPOUT)+'_DOR'+str(DROPOUT_RATIO)+'_fold'+str(TEST_NUM)+'.png', dpi = 300)

def one_hot_label2label_vec(image_data, original_img):
    (h, w, _) = image_data.shape
    label_mat = np.zeros((h, w))
    label_vec = [] 
    for y in range(h):
        for x in range(w):
           #!RGB becuase PIL is used:
           if not (original_img[y][x] == [1,1,1]).all():
               label = np.argmax(image_data[y][x])
               if label >= nb_class:
                   label = 0
               label_vec.append(label)
    return label_vec

def predict(input_dir,TEST_LIST):
    (file_names, inputs) = load_images_test(DIR_TESTS, INPUT_IMAGE_SIZE, END_NUM, TEST_LIST, ORIGINAL_COLOR, ROTATION=False, whole_prediction=DO_WHOLE_PREDICTION)
    print('predict_img_files:', file_names)
    network = UNet(INPUT_IMAGE_SIZE, layer_num = LAYER_NUM, BN=BN_POSITION, nb_class = nb_class)
    model = network.model()
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    print('start prediction... file_names:', file_names, 'len(file_names):', len(file_names), 'inputs:', len(inputs))
    start_time = time.time()
    preds = model.predict(inputs, BATCH_SIZE)
    end_time = time.time()
    for pred in preds:
        print(np.array([np.sum(pred.argmax(axis=2) == i) for i in range(nb_class)]))
        #print(pred)
    print('prediction finished. Calculation time:', end_time - start_time, 'Ave. calc. time', (end_time - start_time)/len(file_names))
    save_images(DIR_OUTPUTS, preds, file_names)

    #Evaluation
    #(gt_files, gt_list) = load_images_train_l(DIR_TEACHERS, INPUT_IMAGE_SIZE, END_NUM, TEST_LIST, TEST_LIST, ORIGINAL_COLOR, ROTATION=False) 
    (gt_files, gt_list) = load_images_train_l(DIR_TEST_LABEL, INPUT_IMAGE_SIZE, END_NUM, TEST_LIST, TEST_LIST, ORIGINAL_COLOR, False, nb_class)
    
    all_gt_label_vec, all_pred_label_vec = [], []
    for i,_ in enumerate(preds):
        print(file_names[i])
        gt_label_vec = one_hot_label2label_vec(gt_list[i], inputs[i])
        pred_label_vec = one_hot_label2label_vec(preds[i], inputs[i])
        all_gt_label_vec += gt_label_vec
        all_pred_label_vec += pred_label_vec
        #print('Confusion_matrix:', confusion_matrix(gt_label_vec, pred_label_vec))

    print("============================================================================")
    print('Confusion_matrix', confusion_matrix(all_gt_label_vec, all_pred_label_vec))
    print(classification_report(all_gt_label_vec, all_pred_label_vec))
    print('P,R,F1 (macro)', precision_recall_fscore_support(all_gt_label_vec, all_pred_label_vec, average="macro"))
    print('P,R,F1 (micro)', precision_recall_fscore_support(all_gt_label_vec, all_pred_label_vec, average="micro"))
    print('P,R,F1 (weighted)', precision_recall_fscore_support(all_gt_label_vec, all_pred_label_vec, average="weighted"))

def predict_dif_line(TEST_LIST):

    DIR_TESTS = os.path.join('./', '1207_S1_'+TEST_DIR+'_labeloriginal_rotation')
    DIR_TEST_LABEL = os.path.join('./', '1207_S1_'+TEST_DIR+'_converted_label_rotation')
    DIR_OUTPUTS = os.path.join('./', LOAD_OUTPUT_DIR)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    (file_names, inputs) = load_images_test(DIR_TESTS, INPUT_IMAGE_SIZE, END_NUM, TEST_LIST, ORIGINAL_COLOR, ROTATION=False, whole_prediction=DO_WHOLE_PREDICTION)
    print('predict_img_files:', file_names)
    network = UNet(INPUT_IMAGE_SIZE, layer_num = LAYER_NUM, BN=BN_POSITION, nb_class = nb_class)
    model = network.model()
    model.load_weights(LOAD_MODEL_DIR)
    print('start prediction... file_names:', file_names, 'len(file_names):', len(file_names), 'inputs:', len(inputs))
    start_time = time.time()
    preds = model.predict(inputs, BATCH_SIZE)
    end_time = time.time()
    for pred in preds:
        print(np.array([np.sum(pred.argmax(axis=2) == i) for i in range(nb_class)]))
        #print(pred)
    print('prediction finished. Calculation time:', end_time - start_time, 'Ave. calc. time', (end_time - start_time)/len(file_names))
    save_images(DIR_OUTPUTS, preds, file_names)

    #Evaluation
    #(gt_files, gt_list) = load_images_train_l(DIR_TEACHERS, INPUT_IMAGE_SIZE, END_NUM, TEST_LIST, TEST_LIST, ORIGINAL_COLOR, ROTATION=False) 
    (gt_files, gt_list) = load_images_train_l(DIR_TEST_LABEL, INPUT_IMAGE_SIZE, END_NUM, TEST_LIST, TEST_LIST, ORIGINAL_COLOR, False, nb_class)
    
    all_gt_label_vec, all_pred_label_vec = [], []
    for i,_ in enumerate(preds):
        print(i, file_names[i])
        gt_label_vec = one_hot_label2label_vec(gt_list[i], inputs[i])
        pred_label_vec = one_hot_label2label_vec(preds[i], inputs[i])
        all_gt_label_vec += gt_label_vec
        all_pred_label_vec += pred_label_vec
        #print('Confusion_matrix:', confusion_matrix(gt_label_vec, pred_label_vec))

    print("============================================================================")
    print('Confusion_matrix', confusion_matrix(all_gt_label_vec, all_pred_label_vec))
    print(classification_report(all_gt_label_vec, all_pred_label_vec))
    print('P,R,F1 (macro)', precision_recall_fscore_support(all_gt_label_vec, all_pred_label_vec, average="macro"))
    print('P,R,F1 (micro)', precision_recall_fscore_support(all_gt_label_vec, all_pred_label_vec, average="micro"))
    print('P,R,F1 (weighted)', precision_recall_fscore_support(all_gt_label_vec, all_pred_label_vec, average="weighted"))


def whole_predict(File_MODEL):
    #File_MODEL = 'model_' + str(SET_FOLD+1) + '.hdf5'
    File_MODEL = File_MODEL

    #DIR_TESTS = os.path.join('./', TEST_DIR+'_all_png')
    #DIR_OUTPUTS = os.path.join('./', TEST_DIR+'_all_predict')
    DIR_TESTS = os.path.join('./', 'test_png')
    DIR_OUTPUTS = os.path.join('./', 'test_predict')
    if not os.path.isdir(DIR_OUTPUTS):
        os.mkdir(DIR_OUTPUTS)
    
    print('DIR_TESTS', DIR_TESTS)

    files = glob.glob(os.path.join(DIR_TESTS+'/*.png'))
    END_NUM = len(files)
    PREDICT_LIST = range(0, END_NUM)
    chuncked_list = list(chunked(PREDICT_LIST, 100))
    print('NUM', END_NUM, 'chuncked list num', len(chuncked_list))


    network = UNet(INPUT_IMAGE_SIZE, layer_num = LAYER_NUM, BN=BN_POSITION, nb_class = nb_class)
    model = network.model()
    #model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    model.load_weights(File_MODEL)
    
    ground_start_time = time.time()
    for l in chuncked_list:
        (file_names, inputs) = load_images_test(DIR_TESTS, INPUT_IMAGE_SIZE, END_NUM, l, ORIGINAL_COLOR, ROTATION, whole_prediction=DO_WHOLE_PREDICTION)
        print('start prediction... file_names:', file_names, 'len(file_names):', len(file_names), 'inputs:', len(inputs))
        start_time = time.time()
        preds = model.predict(inputs, BATCH_SIZE)
        end_time = time.time()
        print('prediction finished. Calculation time:', end_time - start_time, 'Ave. calc. time', (end_time - start_time)/len(file_names))
        save_images(DIR_OUTPUTS, preds, file_names)
    ground_end_time = time.time()
    print('TOTAL_TLIME', (ground_end_time - ground_start_time), ' AVE TIME/img', (ground_end_time - ground_start_time)/(END_NUM-1))


if __name__ == '__main__':
    #args = get_option(EPOCHS)
    #EPOCHS = args.epoch
    
    
    if DO_WHOLE_PREDICTION:
        #print('PARAMETER_DISC', PARAMETER_DISC)
        print('Load model', LOAD_MODEL_DIR)
        
        #File_MODEL = 'model_whole_' + PARAMETER_DISC + '.hdf5'
        TEST_LIST = []
        if DO_TRAIN:
            train(TEST_LIST, 0)
        #whole_predict(File_MODEL)
        whole_predict(LOAD_MODEL_DIR)
        #File_MODEL = 'model_' + str(SET_FOLD+1) + '.hdf5'
        #END_NUM = 14017
        #DIR_TESTS = 'whole_contrast_after'
        #DIR_OUTPUTS = 'whole_contrast_after_predict'
        #PREDICT_LIST = range(1, END_NUM)
        #chuncked_list = list(chunked(PREDICT_LIST, 100))
        #for l in chuncked_list:
        #   predict(DIR_TESTS, l)
    elif TEST_DIF_LINE:
        
        file_index_list = range(0, 4000+1, 10)
        TRAIN_LIST = file_index_list
        TEST_LIST = file_index_list

        predict_dif_line(TEST_LIST)


    else:
        #1つのデータ内でHold-out検証を行う場合と, trainとtestのディレクトリが異なる場合で処理を変える. 
        #1つのディレクトリの中でHold-out検証を行う場合
        if TRAIN_DIR == TEST_DIR:
            file_index_list = range(0, 4000+1, 10) #ここは, 各lineの枚数のmaxを入れる. 大きめでOK
            #file_index_list = range(0, 300+1, 10)
            data_num = len(file_index_list)
            #print(label_files)
            TEST_LIST, TRAIN_LIST = [], []
            for i in range(data_num):
                if i % TEST_SPLIT_NUM == SET_FOLD:
                    TEST_LIST.append(file_index_list[i])
                else:
                    TRAIN_LIST.append(file_index_list[i])
           
            #訓練データを間引く場合. 
            if REDUCTION >= 2:
                TRAIN_LIST = [TRAIN_LIST[i] for i in range(len(TRAIN_LIST)) if i % REDUCTION == 2]
            
            print('TRAIN_LIST', TRAIN_LIST)
            print('TEST_LIST', TEST_LIST)
            File_MODEL = 'model_' + str(SET_FOLD+1) + '.hdf5'

            if DO_TRAIN:
                train(TEST_LIST, TRAIN_LIST, SET_FOLD)
            if DO_TEST:
                predict(DIR_TESTS, TEST_LIST) 
        else:
            file_index_list = range(0, 4000+1, 10)
            TRAIN_LIST = file_index_list
            TEST_LIST = file_index_list
            
            #訓練データを間引く場合. 
            if REDUCTION >= 2:
                TRAIN_LIST = [TRAIN_LIST[i] for i in range(len(TRAIN_LIST)) if i % REDUCTION == 2]
            
            print('TRAIN_LIST', TRAIN_LIST)
            print('TEST_LIST', TEST_LIST)
            

            File_MODEL = 'model.hdf5'

            if DO_TRAIN:
                train(TEST_LIST, TRAIN_LIST, -1)
            if DO_TEST:
                predict(DIR_TESTS, TEST_LIST) 
        """
        for TEST_NUM in range(0,TEST_SPLIT_NUM):
            if SET_FOLD < 0 or TEST_NUM == SET_FOLD:
                File_MODEL = 'model_' + str(TEST_NUM+1) + '.hdf5' 
                TEST_LIST = list(range(1+TEST_NUM*TEST_DELTA,1+(TEST_NUM+1)*TEST_DELTA))
                print('TEST_LIST', TEST_LIST)
                if DO_TRAIN:
                    train(TEST_LIST, TEST_NUM)
                if DO_TEST:
                    predict(DIR_TESTS,TEST_LIST)

        """
        
