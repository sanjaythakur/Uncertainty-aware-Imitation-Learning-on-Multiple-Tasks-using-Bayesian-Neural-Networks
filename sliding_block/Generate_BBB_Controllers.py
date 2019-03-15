from datetime import datetime
import tensorflow as tf, argparse, sys, os, copy, _pickle as pickle

from Sliding_Block import *
#from Validate_BBB_Controllers import validate_BBB_controller

import sys
sys.path.insert(0,'./../')
from Housekeeping import *
from BBBNNRegression import BBBNNRegression
from Dataset import getDemonstrationDataset


def generate_BBB_controller(context_code, window_size, partial_observability, epochs, number_mini_batches, activation_unit, learning_rate, hidden_units, number_samples_variance_reduction, precision_alpha, weights_prior_mean_1,
                     weights_prior_mean_2, weights_prior_deviation_1, weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis):

    configuration_identity = str(context_code) + '_' + str(window_size) + '_' + str(partial_observability) + '_' + 'BBB'

    directory_to_save_tensorboard_data = TENSORBOARD_DIRECTORY + configuration_identity + '/'
    saved_models_during_iterations_bbb = SAVED_MODELS_DURING_ITERATIONS_DIRECTORY + configuration_identity + '/'
    saved_final_model_bbb = SAVED_FINAL_MODEL_DIRECTORY + configuration_identity + '/'
    if not os.path.exists(INPUT_MANIPULATION_DIRECTORY):
        os.makedirs(INPUT_MANIPULATION_DIRECTORY)   
    if not os.path.exists(directory_to_save_tensorboard_data):
        os.makedirs(directory_to_save_tensorboard_data)
    if not os.path.exists(saved_models_during_iterations_bbb):
        os.makedirs(saved_models_during_iterations_bbb)
    if not os.path.exists(saved_final_model_bbb):
        os.makedirs(saved_final_model_bbb)

    start_time = datetime.now()
    moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size = getDemonstrationDataset(all_block_masses= get_sliding_block_context_from_code(context_code=context_code), window_size=window_size,
                                                         partial_observability=partial_observability)
    print(RED('Time taken to generate dataset is ' + str(datetime.now()-start_time)))

    # House-keeping to make data amenable for good training
    mean_x, deviation_x = get_mean_and_deviation(data = moving_windows_x)
    moving_windows_x = NORMALIZE(moving_windows_x, mean_x, deviation_x)

    mean_y, deviation_y = get_mean_and_deviation(data = moving_windows_y)
    moving_windows_y = NORMALIZE(moving_windows_y, mean_y, deviation_y)

    file_name_to_save_input_manipulation_data = INPUT_MANIPULATION_DIRECTORY + configuration_identity + '.pkl'
    normalization_data_to_store = {MEAN_KEY_X: mean_x, DEVIATION_KEY_X: deviation_x, MEAN_KEY_Y:mean_y, DEVIATION_KEY_Y:deviation_y,
                                     DRIFT_PER_TIME_STEP_KEY: drift_per_time_step, MOVING_WINDOWS_X_SIZE_KEY: moving_windows_x_size}
    with open(file_name_to_save_input_manipulation_data, 'wb') as f:
        pickle.dump(normalization_data_to_store, f)
    
    print(GREEN('Creating the BBB based Bayesian NN'))
    BBB_Regressor=BBBNNRegression(number_mini_batches=number_mini_batches, number_features=moving_windows_x.shape[1], number_output_units=moving_windows_y.shape[1], activation_unit=activation_unit, learning_rate=learning_rate,
                                         hidden_units=hidden_units, number_samples_variance_reduction=number_samples_variance_reduction, precision_alpha=precision_alpha,
                                             weights_prior_mean_1=weights_prior_mean_1, weights_prior_mean_2=weights_prior_mean_2, weights_prior_deviation_1=weights_prior_deviation_1,
                                                 weights_prior_deviation_2=weights_prior_deviation_2, mixture_pie=mixture_pie, rho_mean=rho_mean, extra_likelihood_emphasis=extra_likelihood_emphasis)
    print(GREEN('BBB based Bayesian NN created successfully'))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(directory_to_save_tensorboard_data, sess.graph)
        saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=2)
        previous_minimum_loss = sys.float_info.max
        mini_batch_size = int(moving_windows_x.shape[0]/number_mini_batches)
        for epoch_iterator in range(epochs):
            moving_windows_x, moving_windows_y = randomize(moving_windows_x, moving_windows_y)
            ptr = 0
            for mini_batch_iterator in range(number_mini_batches):
                x_batch = moving_windows_x[ptr:ptr+mini_batch_size, :]
                y_batch = moving_windows_y[ptr:ptr+mini_batch_size, :]
                _, loss, summary = sess.run([BBB_Regressor.train(), BBB_Regressor.getMeanSquaredError(), BBB_Regressor.summarize()], feed_dict={BBB_Regressor.X_input:x_batch, BBB_Regressor.Y_input:y_batch})
                sess.run(BBB_Regressor.update_mini_batch_index())
                if loss < previous_minimum_loss:
                    saver.save(sess, saved_models_during_iterations_bbb + 'iteration', global_step=epoch_iterator, write_meta_graph=False)
                    previous_minimum_loss = loss
                ptr += mini_batch_size
                writer.add_summary(summary, global_step=tf.train.global_step(sess, BBB_Regressor.global_step))
            if epoch_iterator % 2500 == 0:
                print(RED('Training progress: ' + str(epoch_iterator) + '/' + str(epochs)))     
        writer.close()
        saver.save(sess, saved_final_model_bbb + 'final', write_state=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--context_code', type=int, help='Contexts to train on', default=0)
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=1)
    parser.add_argument('-po', '--partial_observability', type=str, help='Partial Observability', default='True')
    args = parser.parse_args()

    print(GREEN('Settings are context code ' + str(args.context_code) + ', window size is ' + str(args.window_size) + ', partial observability is ' + str(args.partial_observability)))
    
    generate_BBB_controller(context_code=args.context_code, window_size=args.window_size, partial_observability=str_to_bool(args.partial_observability),
     epochs = 10001, number_mini_batches = 20, activation_unit = 'RELU', learning_rate = 0.001, hidden_units= [90, 30, 10], number_samples_variance_reduction = 25, precision_alpha = 0.01,
       weights_prior_mean_1 = 0., weights_prior_mean_2 = 0., weights_prior_deviation_1 = 0.4, weights_prior_deviation_2 = 0.4, mixture_pie = 0.7, rho_mean = -3.,
        extra_likelihood_emphasis = 10000000000000000.) 