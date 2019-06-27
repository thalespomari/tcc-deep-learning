import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from scipy import interp
import matplotlib.cm as colormap
import time
import os.path
import keras.applications as keras_app
import os.path
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.ticker as ticker


def resnet50(image_shape):
    return keras_app.resnet50.ResNet50(weights='imagenet', input_shape=image_shape, include_top=False, pooling='avg')


def vgg16(image_shape):
    return keras_app.vgg16.VGG16(weights='imagenet', input_shape=image_shape, include_top=False, pooling='avg')


def vgg19(image_shape):
    return keras_app.vgg19.VGG19(weights='imagenet', input_shape=image_shape, include_top=False, pooling='avg')


def inceptionv3(image_shape):
    return keras_app.inception_v3.InceptionV3(weights='imagenet', input_shape=image_shape, include_top=False,
                                              pooling='avg')


def inception_resnet_v2(image_shape):
    return keras_app.inception_resnet_v2.InceptionResNetV2(weights='imagenet', input_shape=image_shape,
                                                           include_top=False, pooling='avg')


def plot_mean_acc(history, path_resultados, test_name, db_name):
    print("Plotando a acurácia")
    figure = plt.gcf()
    figure.set_size_inches(20, 8)
    ax = plt.subplot()
    plt.title('Acurácia')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    colors = iter(colormap.gist_rainbow(np.linspace(0, 1, len(history))))
    i = 1
    for h in history:
        color=next(colors)
        plt.plot(h.history['acc'], label='Treino '+str(i), color=color, linestyle = 'solid')
        plt.plot(h.history['val_acc'], label='Teste '+str(i), color=color, linestyle = 'dotted')
        i += 1
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.savefig(path_resultados + '/' + test_name + '_' + db_name + "_acc.jpg")
    plt.cla()
    plt.clf()


def plot_mean_loss(history, path_resultados, test_name, db_name):
    print("Plotando a Loss")
    figure = plt.gcf()
    figure.set_size_inches(20, 8)
    ax = plt.subplot()
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Época')
    colors = iter(colormap.gist_rainbow(np.linspace(0, 1, len(history))))
    i = 1
    for h in history:
        color=next(colors)
        plt.plot(h.history['loss'], label='Treino '+str(i), color=color, linestyle = 'solid')
        plt.plot(h.history['val_loss'], label='Teste '+str(i), color=color, linestyle = 'dotted')
        i += 1
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.savefig(path_resultados + '/' + test_name + '_' + db_name + "_loss.jpg")
    plt.cla()
    plt.clf()


# def svm_svc_rep(db_path, db_name, base_model, cnn_name, test_name, image_shape):
#     np.random.seed(1)
#     image_dir = db_path
#     current_dir = os.getcwd()
#     os.chdir(image_dir)  # selecting the parent folder with sub-folders

#     # Get number of samples per family
#     list_classes = sorted(os.listdir(os.getcwd()), key=str.lower)  # vector of strings with family names
#     number_samples_per_class = []  # No. of samples per family
#     for i in range(len(list_classes)):
#         os.chdir(list_classes[i])
#         len1 = len(glob.glob('*.png'))  # assuming the images are stored as 'png'
#         number_samples_per_class.append(len1)
#         os.chdir('..')
#     total_number_samples = np.sum(number_samples_per_class)  # total number of all samples

#     # Compute the labels
#     y = np.zeros(total_number_samples)
#     pos = 0
#     label = 0
#     for i in number_samples_per_class:
#         print("Label:%2d\tFamilia: %15s\tNumero de Imagens: %d" % (label, list_classes[label], i))
#         for j in range(i):
#             y[pos] = label
#             pos += 1
#         label += 1
#     num_classes = label

#     # Compute the features
#     width, height, channels = image_shape
#     X = np.zeros((total_number_samples, width, height, channels))
#     cnt = 0
#     list_paths = []  # List of image paths
#     mt_label_img = np.zeros((total_number_samples, 1), dtype=np.object_)
#     print("Processando imagens ...")
#     for i in range(len(list_classes)):
#         for img_file in glob.glob(list_classes[i] + '/*.png'):
#             print("[%d] Processando imagem: %s" % (cnt, img_file))
#             list_paths.append(os.path.join(os.getcwd(), img_file))
#             img = image.load_img(img_file, target_size=image_shape[:-1])
#             x = image.img_to_array(img)
#             x = np.expand_dims(x, axis=0)
#             x = preprocess_input(x)
#             X[cnt] = x
#             mt_label_img[cnt] = img_file
#             cnt += 1
#     print("Imagens processadas: %d" % cnt)

#     os.chdir(current_dir)

#     # Encoding classes (y) into integers (y_encoded) and then generating one-hot-encoding (Y)
#     encoder = LabelEncoder()
#     encoder.fit(y)
#     y_encoded = encoder.transform(y)
#     Y = np_utils.to_categorical(y_encoded)

#     if not os.path.exists('./features'):
#         os.mkdir('./features')

#     filename = './features/' + db_name + "-" + test_name + '-svm-svc.npy'
#     if os.path.exists(filename):
#         print("Carregando features extraidas de %s ..." % filename)
#         extracted_features = np.load(filename)
#     else:
#         print("Extraindo features ...")
#         extracted_features = base_model.predict(X)
#         print("Salvando features extraidas em %s ..." % filename)
#         np.save(filename, extracted_features)

#     # Create stratified k-fold subsets
#     kfold = 5  # no. of folds
#     skf = StratifiedKFold(kfold, shuffle=True, random_state=1)
#     skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
#     cnt = 0
#     for index in skf.split(X, y):
#         skfind[cnt] = index
#         cnt += 1

#     # Training top_model and saving min training loss weights
#     conf_mat = np.zeros((len(list_classes), len(list_classes)))  # Initializing the Confusion Matrix
#     path_resultados = "./resultados/" + test_name + "/" + db_name
#     if not os.path.exists(path_resultados):
#         os.makedirs(path_resultados)
#     testing_results = path_resultados + "/" + test_name + "_resultado.txt"

#     tprs = []
#     aucs = []
#     fold_cnt = 1
#     mean_fpr = np.linspace(0, 1, 100)

#     for i in range(kfold):
#         train_indices = skfind[i][0]
#         test_indices = skfind[i][1]
#         X_train = extracted_features[train_indices]
#         y_train = y[train_indices]
#         X_test = extracted_features[test_indices]
#         y_test = y[test_indices]
#         img_fold_teste = mt_label_img[test_indices]

#         top_model = svm.SVC(gamma='scale', probability=True)
#         probas_ = top_model.fit(X_train, y_train).predict_proba(X_test) # Training
#         y_pred = top_model.predict(X_test)  # Testing

#         fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#         plt.plot(fpr, tpr, lw=1, alpha=0.3,
#                  label='ROC fold %d (AUC = %0.2f)' % (fold_cnt, roc_auc))
#         fold_cnt += 1

#         contador = 0
#         testes_salvos = ""
#         if os.path.isfile(testing_results):
#             with open(testing_results, "r") as resultado:
#                 testes_salvos = resultado.read()

#         with open(testing_results, "w") as text_file:
#             text_file.write(testes_salvos)
#             text_file.write("\n\n\n[%d] Test acurracy: %.4f\n" % (i, accuracy_score(y_test, y_pred)))
#             for pred in y_pred:
#                 if pred == 0:
#                     text_file.write("\n%s = Normal" % img_fold_teste[contador])
#                 else:
#                     text_file.write("\n%s = Splicing" % img_fold_teste[contador])
#                 contador += 1
#         print("[%d] Test acurracy: %.4f" % (i, accuracy_score(y_test, y_pred)))
#         cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix for this fold
#         conf_mat = conf_mat + cm  # Compute global confusion matrix

#     # Calculando a acuracia media
#     avg_acc = np.trace(conf_mat) / np.sum(conf_mat)
#     print("Acuracia media: %.4f" % avg_acc)

#     acc_results = './resultados/acuracia-total.txt'
#     acc_salva = ''
#     if os.path.isfile(acc_results):
#         with open(acc_results, "r") as acc_result:
#             acc_salva = acc_result.read()

#     with open(acc_results, "w+") as text_file:
#         text_file.write(acc_salva)
#         text_file.write("%s, %s, %.4f\n" % (cnn_name, db_name, avg_acc))

#     # Plotando a curva ROC
#     print("Plotando a curva ROC")
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Falso Positivo')
#     plt.ylabel('Verdadeiro Positivo')
#     plt.legend(loc="lower right")
#     plt.savefig(path_resultados + '/' + test_name + '_' + db_name + "_roc.jpg")
#     plt.cla()
#     plt.clf()

#     # Plotando a matriz de confusao
#     print("Plotando a matriz de confusao")
#     conf_mat = conf_mat.T  # since rows and cols are interchangeable
#     conf_mat_norm = conf_mat / number_samples_per_class  # Normalizing the confusion matrix
#     conf_mat = np.around(conf_mat_norm, decimals=2)  # rounding to display in figure
#     figure = plt.gcf()
#     figure.set_size_inches(8, 6)
#     plt.imshow(conf_mat, interpolation='nearest')
#     for row in range(len(list_classes)):
#         for col in range(len(list_classes)):
#             plt.annotate(str(conf_mat[row][col]), xy=(col, row), ha='center', va='center')
#     plt.xticks(range(len(list_classes)), list_classes, rotation=90, fontsize=10)
#     plt.yticks(range(len(list_classes)), list_classes, fontsize=10)
#     plt.colorbar()
#     if not os.path.exists(path_resultados):
#         os.makedirs(path_resultados)
#     plt.savefig(path_resultados + '/' + test_name + '_' + db_name + "_matrix.jpg")
#     plt.cla()
#     plt.clf()


def svm_svc_scratch(db_path, db_name, cnn_name, test_name, image_shape):
    np.random.seed(1)
    image_dir = db_path
    current_dir = os.getcwd()
    os.chdir(image_dir)  # selecting the parent folder with sub-folders

    # Get number of samples per family
    list_classes = sorted(os.listdir(os.getcwd()), key=str.lower)  # vector of strings with family names
    number_samples_per_class = []  # No. of samples per family
    for i in range(len(list_classes)):
        os.chdir(list_classes[i])
        len1 = len(glob.glob('*.png'))  # assuming the images are stored as 'png'
        number_samples_per_class.append(len1)
        os.chdir('..')
    total_number_samples = np.sum(number_samples_per_class)  # total number of all samples

    # Compute the labels
    y = np.zeros(total_number_samples)
    pos = 0
    label = 0
    for i in number_samples_per_class:
        print("Label:%2d\tFamilia: %15s\tNumero de Imagens: %d" % (label, list_classes[label], i))
        for j in range(i):
            y[pos] = label
            pos += 1
        label += 1
    num_classes = label

    # Compute the features
    width, height, channels = image_shape
    X = np.zeros((total_number_samples, width, height, channels))
    cnt = 0
    list_paths = []  # List of image paths
    mt_label_img = np.zeros((total_number_samples, 1), dtype=np.object_)
    print("Processando imagens ...")
    for i in range(len(list_classes)):
        for img_file in glob.glob(list_classes[i] + '/*.png'):
            print("[%d] Processando imagem: %s" % (cnt, img_file))
            list_paths.append(os.path.join(os.getcwd(), img_file))
            img = image.load_img(img_file, target_size=image_shape[:-1])
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            X[cnt] = x
            mt_label_img[cnt] = img_file
            cnt += 1
    print("Imagens processadas: %d" % cnt)

    os.chdir(current_dir)

    # Encoding classes (y) into integers (y_encoded) and then generating one-hot-encoding (Y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    Y = np_utils.to_categorical(y_encoded)

    if not os.path.exists('./features'):
        os.mkdir('./features')

    # Create stratified k-fold subsets
    kfold = 5  # no. of folds
    skf = StratifiedKFold(kfold, shuffle=True, random_state=1)
    skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
    cnt = 0
    for index in skf.split(X, y):
        skfind[cnt] = index
        cnt += 1

    path_resultados = "./resultados/" + test_name + "/" + db_name
    if not os.path.exists(path_resultados):
        os.makedirs(path_resultados)
    testing_results = path_resultados + "/" + test_name + "_resultado.txt"

    base_model = ''

    if str.lower(cnn_name) == 'resnet50':
        base_model = keras_app.resnet50.ResNet50(weights=None, input_shape=image_shape, include_top=True,
                                                 classes=num_classes)
    elif str.lower(cnn_name) == 'vgg16':
        base_model = keras_app.vgg16.VGG16(weights=None, input_shape=image_shape, include_top=True,
                                           classes=num_classes)
    elif str.lower(cnn_name) == 'vgg19':
        base_model = keras_app.vgg19.VGG19(weights=None, input_shape=image_shape, include_top=True,
                                           classes=num_classes)
    elif str.lower(cnn_name) == 'inceptionv3':
        base_model = keras_app.inception_v3.InceptionV3(weights=None, input_shape=image_shape, include_top=True,
                                                        classes=num_classes)
    elif str.lower(cnn_name) == 'inceptionresnetv2':
        base_model = keras_app.inception_resnet_v2.InceptionResNetV2(weights=None, input_shape=image_shape,
                                                                     include_top=True, classes=num_classes)

    base_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    filename = './weights/' + db_name + '-' + test_name + '-svm-svc_weights.npy'
    if os.path.exists(filename):
        print("Loading initial weights from %s ..." % (filename))
        init_weights = np.load(filename)
    else:
        print("Generating initial weigths ...")
        init_weights = base_model.get_weights()
        print("Saving initial weights into %s ..." % (filename))
        np.save(filename, init_weights)

    num_epochs = 500
    history = []

    path_checkpoint = './weights/' + db_name + '-' + test_name + '-svm-svc_weights.h5'

    checkpointer = ModelCheckpoint(filepath=path_checkpoint,
                                   monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True,
                                   mode='auto')
    early_stopping = EarlyStopping(verbose=1, patience=100, monitor='val_loss')

    callbacks_list = [checkpointer, early_stopping]
    conf_mat = np.zeros((len(list_classes), len(list_classes)))
    history = []
    tprs = []
    aucs = []
    fold_cnt = 1
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(kfold):
        train_indices = skfind[i][0]
        test_indices = skfind[i][1]
        X_train = X[train_indices]
        Y_train = Y[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        Y_test = Y[test_indices]
        y_test = y[test_indices]
        img_fold_teste = mt_label_img[test_indices]

        base_model.set_weights(init_weights)

        start = time.time()
        h = base_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=num_epochs,
                      batch_size=32, verbose=1, callbacks=callbacks_list)
        end = time.time()
        history.append(h)

        y_prob = base_model.predict(X_test, verbose=1)  # Testing
        y_pred = np.argmax(y_prob, axis=1)
#         print(len(y_prob))
#         print(y_prob)
#         print(len(y_prob[:,0]))
#         print(y_prob[:,0])
        
        print("[%d] Test acurracy: %.4f (%.4f s)" % (i, accuracy_score(y_test, y_pred), end - start))

        cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix for this fold
        conf_mat = conf_mat + cm  # Compute global confusion matrix

        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (fold_cnt, roc_auc))
        fold_cnt += 1

        contador = 0
        testes_salvos = ""
        if os.path.isfile(testing_results):
            with open(testing_results, "r") as resultado:
                testes_salvos = resultado.read()

        with open(testing_results, "w") as text_file:
            text_file.write(testes_salvos)
            text_file.write("\n\n\n[%d] Test acurracy: %.4f\n" % (i, accuracy_score(y_test, y_pred)))
            for pred in y_pred:
                if pred == 0:
                    text_file.write("\n%s = Normal" % img_fold_teste[contador])
                else:
                    text_file.write("\n%s = Splicing" % img_fold_teste[contador])
                contador += 1

    # Calculando a acuracia media
    avg_acc = np.trace(conf_mat) / np.sum(conf_mat)
    print("Acuracia media: %.4f" % avg_acc)

    acc_results = './resultados/acuracia-total-scratch.txt'
    acc_salva = ''
    if os.path.isfile(acc_results):
        with open(acc_results, "r") as acc_result:
            acc_salva = acc_result.read()

    with open(acc_results, "w+") as text_file:
        text_file.write(acc_salva)
        text_file.write("%s, %s, %.4f\n" % (cnn_name, db_name, avg_acc))
    
        
    #Plotando a curva ROC
    print("Plotando a curva ROC")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.legend(loc="lower right")
    plt.savefig(path_resultados + '/' + test_name + '_' + db_name + "_roc.jpg")
    plt.cla()
    plt.clf()
    
    # Plotando a matriz de confusao
    print("Plotando a matriz de confusao")
    conf_mat = conf_mat.T  # since rows and cols are interchangeable
    conf_mat_norm = conf_mat / number_samples_per_class  # Normalizing the confusion matrix
    conf_mat = np.around(conf_mat_norm, decimals=2)  # rounding to display in figure
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.imshow(conf_mat, interpolation='nearest')
    for row in range(len(list_classes)):
        for col in range(len(list_classes)):
            plt.annotate(str(conf_mat[row][col]), xy=(col, row), ha='center', va='center')
    plt.xticks(range(len(list_classes)), list_classes, rotation=90, fontsize=10)
    plt.yticks(range(len(list_classes)), list_classes, fontsize=10)
    plt.colorbar()
    if not os.path.exists(path_resultados):
        os.makedirs(path_resultados)
    plt.savefig(path_resultados + '/' + test_name + '_' + db_name + "_matrix.jpg")
    plt.cla()
    plt.clf()

    # Plotando a acurácia
    plot_mean_acc(history, path_resultados, test_name, db_name)


    # Plotando a Loss
    plot_mean_loss(history, path_resultados, test_name, db_name)
    
    plt.cla()
    plt.clf()


# if __name__ == "__main__":

#     #python3 cnn.py --dbpath '../database_tcc/database-DSI/normal' --dbname 'DSI_RGB' --cnnname 'resnet50' --testname 'ResNet50_rep_learning'

#     parser = argparse.ArgumentParser(description='Running ConvNet Script')
#     parser.add_argument('--dbpath', metavar='path/to/database', required=True,
#                         help='Path to database.')
#     parser.add_argument('--dbname', metavar='DB_Name', required=True,
#                         help='Name of database.')
#     parser.add_argument('--cnnname', metavar='ResNet50, VGG16, VGG19 or InceptionV3', required=True,
#                         help='CNN to extract the features.')
#     parser.add_argument('--testname', metavar='DB_Name', required=True,
#                         help='Name of test.')

#     args = parser.parse_args()

#     if str.lower(args.cnnname) == 'resnet50':
#         args.cnname = 'ResNet50'
#         image_shape = (224, 224, 3)
#         base_model = resnet50(image_shape)
#     elif str.lower(args.cnnname) == 'vgg16':
#         args.cnname = 'VGG19'
#         image_shape = (224, 224, 3)
#         base_model = vgg16(image_shape)
#     elif str.lower(args.cnnname) == 'vgg19':
#         args.cnname = 'VGG19'
#         image_shape = (224, 224, 3)
#         base_model = vgg19(image_shape)
#     elif str.lower(args.cnnname) == 'inceptionv3':
#         args.cnname = 'InceptionV3'
#         image_shape = (299, 299, 3)
#         base_model = inceptionv3(image_shape)
#     elif str.lower(args.cnnname) == 'inceptionresnetv2':
#         args.cnname = 'InceptionResNetV2'
#         image_shape = (299, 299, 3)
#         base_model = inception_resnet_v2(image_shape)
#     else:
#         print('Invalid CNN name, please fill with one of "resnet50, vgg16, vgg19, inceptionv3 or InceptionResNetV2".')
#         exit()

#     svm_svc_rep(args.dbpath, args.dbname, base_model, args.cnnname, args.testname)

#     exit()
