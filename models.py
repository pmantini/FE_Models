from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.callbacks import TensorBoard
from FE_DB_Ops import DB_Ops
from keras.optimizers import Adam

import random
from Hyper_Setup import log_file

import numpy as np

import logging
logging = logging.getLogger("main")


class FEModel:
    def __init__(self, name, req_args, opt_args):
        self.name = name
        self.req_args = req_args
        self.opt_args = opt_args

    def get_args(self):
        return {"required": self.req_args, "optional": self.opt_args}

    def do_train(self, args):
        pass

    def do_predict(self):
        pass


class points(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = []
        self.opt_args = ['company_list', 'black_list', 'db_folder', 'ouput_dir', 'days_per_sample', 'batch_size', 'epochs', 'epoch_per_batch', 'training_exclude_days', "tensorboard_dir"]
        FEModel.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None

        self.days_per_sample = None
        self.batch_size = None
        self.epochs = None
        self.epochs_per_batch = None

        self.db = None

        self.input_shape = None
        self.points_model = None


    def do_init(self, args):
        self.company_list = args["company_list"] if "company_list" in args.keys() else "companylist.csv"
        self.black_list = args["black_list"] if "black_list" in args.keys() else "blacklist.csv"
        self.db_folder = args["db_folder"] if "db_folder" in args.keys() else "Databases/"
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.days_per_sample = int(args["days_per_sample"]) if "days_per_sample" in args.keys() else 200
        self.batch_size = args["batch_size"] if "batch_size" in args.keys() else 30
        self.epochs = int(args["epochs"]) if "epochs" in args.keys() else 10
        self.epochs_per_batch = args["epochs_per_batch"] if "epochs_per_batch" in args.keys() else 1

        self.training_exclude_days = args["training_exclude_days"] if "training_exclude_days" in args.keys() else 30
        self.tensorboard_dir = args["tensorboard_dir"] if "tensorboard_dir" in args.keys() else "model_logs/"+self.name

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops(company_list_file = self.company_list, black_list_file = self.black_list, db_folder = self.db_folder)

        companies_count = self.db.get_companies_count()
        self.input_shape = (companies_count, self.days_per_sample, 1)


        self.points_model = self.model_init(self.input_shape)
        self.tensorboard = TensorBoard(log_dir=self.tensorboard_dir)

    def model_init(self, input_shape):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))

        model.add(Dense(491, activation='relu'))
        model.add(Dense(1, activation = 'sigmoid'))
        # model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
        #                  input_shape=input_shape))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(64, (3, 3)))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(128, (3, 3)))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(256, (3, 3)))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(1000))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Dense(491, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))

        logging.info(model.to_json())
        model.summary()

        return model


    def compile_model(self, model):
        optimizer = Adam(0.0002, 0.5)
        logging.info("Compiling...")
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model


    def generate_data(self, exclude_rows_from_end = 30, train_partiion = .8):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows() - exclude_rows_from_end

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch = self.db.get_values_company(company_sym=k)[:-exclude_rows_from_end]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        total_samples_avail  = max_items - self.days_per_sample
        train_sample_size = int(train_partiion*total_samples_avail)

        random_samples = random.sample(range(0, total_samples_avail), total_samples_avail)

        random_samples = [k for k in range(total_samples_avail)]

        train_samples = random_samples[:train_sample_size]
        eval_samples = random_samples[train_sample_size:]



        if len(eval_samples) < self.days_per_sample:
            logging.error("Not enough samples to create valuation set, only %s available, require %s: Exiting", len(eval_samples), self.days_per_sample)
            exit()

        train_iter = 0
        eval_iter = 0

        epoch_iter = 0
        batch_count = 0

        x_train, y_train = None, None
        x_eval, y_eval = None, None
        while True:

            if epoch_iter >= self.epochs:
                logging.info("Max epochs reached: Exiting")
                raise StopIteration

            if train_iter >= len(train_samples):
                epoch_iter += 1
                train_iter = 0

            eval_iter = 0 if eval_iter >= len(eval_samples) else eval_iter

            if x_train is None:

                temp_sample = values[:, train_samples[train_iter]:train_samples[train_iter] + self.days_per_sample + 1]

                x_train = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                y_train = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                y_train[np.isnan(y_train)] = 0
                y_train[np.isinf(y_train)] = 0
                y_train = 1 if np.mean(y_train) > 0 else 0

                temp_sample = values[:, eval_samples[eval_iter]:eval_samples[eval_iter] + self.days_per_sample + 1]

                x_eval = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                y_eval = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                y_eval[np.isnan(y_eval)] = 0
                y_eval[np.isinf(y_eval)] = 0
                y_eval = 1 if np.mean(y_eval) > 0 else 0

                train_iter += 1
                eval_iter += 1
                batch_count += 1

            else:
                if batch_count >= self.batch_size:
                    x_train = x_train.reshape(self.batch_size, total_companies, self.days_per_sample)
                    y_train = y_train.reshape(self.batch_size, 1)

                    x_eval = x_eval.reshape(self.batch_size, total_companies, self.days_per_sample)
                    y_eval = y_eval.reshape(self.batch_size, 1)

                    yield x_train, y_train, x_eval, y_eval
                    x_train, y_train, x_eval, y_eval = None, None, None, None
                    batch_count = 0
                    train_iter += 1
                    eval_iter += 1

                    continue

                temp_sample = values[:, train_samples[train_iter]:train_samples[train_iter] + self.days_per_sample + 1]
                temp_samplex = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                x_train = np.vstack((x_train, temp_samplex))

                temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                temp_sampley[np.isnan(temp_sampley)] = 0
                temp_sampley[np.isinf(temp_sampley)] = 0
                temp_sampley = 1 if np.mean(temp_sampley) > 0 else 0
                y_train = np.append(y_train, temp_sampley)

                temp_sample = values[:, eval_samples[eval_iter]:eval_samples[eval_iter] + self.days_per_sample + 1]
                temp_samplex = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                x_eval = np.vstack((x_eval, temp_samplex))

                temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                temp_sampley[np.isnan(temp_sampley)] = 0
                temp_sampley[np.isinf(temp_sampley)] = 0
                temp_sampley = 1 if np.mean(temp_sampley) > 0 else 0
                y_eval = np.append(y_eval, temp_sampley)

                batch_count += 1
                train_iter += 1
                eval_iter += 1



    def do_train(self):
        self.points_model = self.compile_model(self.points_model)

        #setting training data to everything except last n days
        data = self.generate_data(exclude_rows_from_end=self.training_exclude_days)

        while True:
            try:
                x_t, y_t, x_e, y_e = next(data)

                i = 0
                for k in x_t:
                    j = 0
                    for coulmn in k:
                        max_value = np.max(coulmn)
                        if max_value:
                            x_t[i, j] = x_t[i, j] / max_value
                        j += 1
                    i += 1

                x_t = x_t.reshape(self.batch_size, self.db.get_companies_count(), self.days_per_sample, 1)
                y_t = y_t.reshape(self.batch_size, -1)

                i = 0
                for k in x_e:
                    j = 0
                    for coulmn in k:
                        max_value = np.max(coulmn)
                        if max_value:
                            x_e[i, j] = x_e[i, j] / max_value
                        j += 1
                    i += 1

                x_e = x_e.reshape(self.batch_size, self.db.get_companies_count(), self.days_per_sample, 1)
                y_e = y_e.reshape(self.batch_size, -1)

                self.points_model.fit(x_t, y_t, batch_size=self.batch_size, epochs = self.epochs_per_batch, validation_data=(x_e, y_e), callbacks=[self.tensorboard])
            except StopIteration:
                #Save Model
                logging.info("End Training!")
                break








class model2:

    def __init__(self):
        self.name = model2