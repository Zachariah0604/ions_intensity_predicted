import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys

import utils as uti
from tensorflow.python.ops import rnn, rnn_cell 
class WGRAN(object):
    def __init__(self, _lambda,mse_proportion,linear_initialization,lambda_l2_reg,
                 gan_input_size_G,gan_input_size_D,
                 D_hidden_size,learning_rate,
                 seq_length,seq2seq_input_dim,seq2seq_output_dim,
                 lstm_layer_num,lstm_cell_size):
        self.keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
       
        self._lambda=_lambda
        self.mse_proportion=mse_proportion
        if linear_initialization=='None':
            self.linear_initialization=None
        else:
            self.linear_initialization=linear_initialization

        self.lambda_l2_reg=lambda_l2_reg
               
        self.gan_input_size_G=gan_input_size_G
        self.gan_input_size_D=gan_input_size_D
        self.D_hidden_size = D_hidden_size
        self.learning_rate = learning_rate

        self.seq_length=seq_length
        self.seq2seq_input_dim=seq2seq_input_dim
        self.seq2seq_output_dim=seq2seq_output_dim

        self.lstm_layer_num=lstm_layer_num
        self.lstm_cell_size=lstm_cell_size
       
        self._create_gan_model()


    def generator(self):
        
        
        self.encoder_input = tf.placeholder(tf.float32, shape=(None, self.seq2seq_input_dim), name="encoder_input")
        _input=uti._linear('Generator.in_layer.Linear',self.seq2seq_input_dim,self.lstm_cell_size,self.encoder_input,initialization=self.linear_initialization) 

        decoder_input = tf.zeros_like(self.encoder_input, dtype=np.float32, name="GO") + self.encoder_input
       

        
        cells = []
        for i in range(self.lstm_layer_num):
            with tf.variable_scope('Generator.LSTM_{}'.format(i)):
                cells.append(rnn_cell.DropoutWrapper(cell=rnn_cell.BasicLSTMCell(self.lstm_cell_size,forget_bias=1.0,state_is_tuple=True), input_keep_prob=1.0, output_keep_prob=self.keep_prob))
        cell = rnn_cell.MultiRNNCell(cells)

        decoder_output, decoder_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
            [_input],
            [decoder_input],
            cell
           
          
        )


        pred =uti._linear('Generator.out_layer.Linear',self.lstm_cell_size,self.seq2seq_output_dim,decoder_output[0],initialization=self.linear_initialization)
        pred=tf.nn.relu(pred)
        tf.add_to_collection('pred_network', pred)
        return pred

    def ReLULayer(self,name, n_in, n_out, inputs):
        output =  uti._linear(
            name+'.Linear',
            n_in,
            n_out,
            inputs,
            initialization='he'
        )
        output =tf.nn.relu(output)
        return output


    def discriminator(self,_input):
       
        output = self.ReLULayer('Discriminator.1', self.seq2seq_output_dim, self.D_hidden_size, _input)
        output = self.ReLULayer('Discriminator.2', self.D_hidden_size, self.D_hidden_size, output) 
        
        output = self.ReLULayer('Discriminator.3', self.D_hidden_size, self.D_hidden_size, output)
        
        output = uti._linear('Discriminator.4', self.D_hidden_size, 1, output)
        
        return output      
    def _create_gan_model(self):
        
        with tf.variable_scope('Generator'):
            self.fake_data = self.generator()
        with tf.variable_scope('Discriminator'):
            self.real_data = tf.placeholder(tf.float32, [None,self.gan_input_size_D],name='D_x')
            self.disc_real = self.discriminator(self.real_data)
            self.disc_fake = self.discriminator(self.fake_data)

        self.loss_d = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        #self.loss_g = (1-self.mse_proportion)*(-tf.reduce_mean(self.disc_fake))+self.mse_proportion*self.seq2seq_loss(self.real_data,self.fake_data)
        self.loss_g = -tf.reduce_mean(self.disc_fake)+self.seq2seq_loss(self.real_data,self.fake_data)

       
        alpha = tf.random_uniform(
        shape=[self.batch_size,self.seq2seq_output_dim], 
        minval=0.,
        maxval=1.
        )
        differences = self.fake_data - self.real_data
        interpolates = self.real_data + (alpha*differences)
        self.disc_interpolates=self.discriminator(interpolates)
        gradients = tf.gradients(self.disc_interpolates, [interpolates])
        print(gradients)
        
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        self.loss_d += gradient_penalty*self._lambda


        tf.summary.scalar('d_loss', self.loss_d)
        tf.summary.scalar('g_loss', self.loss_g)
        tf.add_to_collection('loss_d', self.loss_d )
        tf.add_to_collection('loss_g', self.loss_g )

        
      
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator')

      
        print("Generator params:")
        for var in self.g_params:
            print("\t{}\t{}".format(var.name, var.get_shape()))
        print("Discriminator params:")
        for var in self.d_params:
            print("\t{}\t{}".format(var.name, var.get_shape()))

        with tf.name_scope('discriminator-train'):
            self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.5,beta2=0.9).minimize(self.loss_d, var_list=self.d_params)
            tf.add_to_collection('disc_train_op', self.disc_train_op )
        with tf.name_scope('generator-train'):
            if(len(self.g_params)>0):
                self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.5,beta2=0.9).minimize(self.loss_g, var_list=self.g_params)
            else:
                self.gen_train_op = tf.no_op()

            tf.add_to_collection('gen_train_op',self.gen_train_op )    
    def seq2seq_loss(self,real,fake):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(fake, [-1], name='reshape_pred')],
            [tf.reshape(real, [-1], name='reshape_target')],
            [tf.ones([self.batch_size*self.seq2seq_output_dim], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
        )
        _loss = tf.div(
            tf.reduce_sum(losses),
            tf.cast(self.batch_size*self.seq2seq_output_dim,tf.float32),
            )
        tf.summary.scalar('seq2seq_loss',_loss)
        return _loss
    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    
class seq2seq:
    def __init__(self, args):
        self.feature_size = args.feature_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.label_size=args.label_size
        self.learning_rate=args.learning_rate
        #self.max_time=20
        self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
        #self.batch_size=16
        self.encoder_inputs_index = tf.placeholder(tf.int32, [None, None],name='encoder_inputs_index')
        self.encoder_sequence_length = tf.placeholder(tf.int32, [None],name='encoder_sequence_length')
        self.decoder_targets = tf.placeholder(tf.float32, [None, None, self.label_size],name='decoder_targets')
        self.seq2seq_model()

    def seq2seq_model(self):
        self.features = tf.placeholder(tf.float32, [None,self.feature_size],name='ions_feature')
        self.encoder_inputs = tf.nn.embedding_lookup(self.features, self.encoder_inputs_index)

        encoder_fw_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
        encoder_bw_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
        encoder_fw_cell = tf.contrib.rnn.DropoutWrapper(encoder_fw_cell,output_keep_prob=self.output_keep_prob)
        encoder_bw_cell = tf.contrib.rnn.DropoutWrapper(encoder_bw_cell,output_keep_prob=self.output_keep_prob)
        #muti_encoder_fw_cell=tf.contrib.rnn.MultiRNNCell([encoder_fw_cell]*self.num_layers)
        #muti_encoder_bw_cell=tf.contrib.rnn.MultiRNNCell([encoder_bw_cell]*self.num_layers)
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell,
                                            cell_bw=encoder_bw_cell,
                                            inputs=self.encoder_inputs,
                                            sequence_length=self.encoder_sequence_length,
                                            dtype=tf.float32, time_major=True)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)#(m,b,hidden)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )
       

        # Decoder
        decoder_lengths = self.encoder_sequence_length
        decoder_cell = LSTMCell(2*self.hidden_size)
        encoder_max_time, batch_size = tf.unstack(tf.shape(self.encoder_inputs_index))

        W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.label_size], 0, 0.1), dtype=tf.float32)
        b = tf.Variable(tf.zeros([self.label_size]), dtype=tf.float32)
        #assert EOS == 1 and PAD == 0

        eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
        pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

        eos_step_embedded = tf.nn.embedding_lookup(self.features, eos_time_slice)
        pad_step_embedded = tf.nn.embedding_lookup(self.features, pad_time_slice)


        def loop_fn_initial():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = eos_step_embedded
            initial_cell_state = encoder_final_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)
        # (time, previous_cell_output, previous_cell_state, previous_loop_state) -> 
        #     (elements_finished, input, cell_state, output, loop_state).
        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

            def get_next_input():
                output_logits = tf.add(tf.matmul(previous_output, W), b) # projection layer
                # [batch_size, vocab_size]
                prediction = tf.argmax(output_logits, axis=1)
                next_input = tf.nn.embedding_lookup(self.features, prediction)
                # [batch_size, input_embedding_size]
                return next_input
            
            elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                          # defining if corresponding sequence has ended

            finished = tf.reduce_all(elements_finished) # -> boolean scalar
            input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
            # input shape [batch_size,input_embedding_size]
            state = previous_state
            output = previous_output
            loop_state = None

            return (elements_finished, 
                    input,
                    state,
                    output,
                    loop_state)
        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:    # time == 0
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
        decoder_outputs =decoder_outputs_ta.stack()
        decoder_outputs_flat = tf.reshape(decoder_outputs, [-1, 2*self.hidden_size])
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
        decoder_logits = tf.transpose(tf.reshape(decoder_logits_flat, [self.max_time, self.batch_size, self.label_size]),(1,0,2))
        self.decoder_prediction = tf.reshape(decoder_logits, [self.batch_size*self.max_time, self.label_size])
        #_,batch_size,_=tf.unstack(tf.shape(encoder_outputs))
        ##eos_step_feature = tf.ones([batch_size,self.feature_size], dtype=tf.float32, name='EOS')
        #eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')*2
        #eos_step_feature = tf.nn.embedding_lookup(self.features, eos_time_slice)
        ## pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
        ## pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)
        #pad_step_feature = tf.zeros([self.batch_size, self.hidden_size*2+self.feature_size],dtype=tf.float32)

     
        #def initial_fn():
        #    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
        #    #initial_input = tf.concat((eos_step_feature, encoder_outputs[0]), 1)#(b,(f+hidden))
        #    initial_input = eos_step_feature
        #    return initial_elements_finished, initial_input

        #def sample_fn(time, outputs, state):
        #    # 选择logit最大的下标作为sample
        #    print("outputs", outputs)
        #    # output_logits = tf.add(tf.matmul(outputs, self.slot_W), self.slot_b)
        #    # print("slot output_logits: ", output_logits)
        #    # prediction_id = tf.argmax(output_logits, axis=1)
        #    prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
        #    return prediction_id

        #def next_inputs_fn(time, outputs, state, sample_ids):
        #    # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
        #    pred_feature = tf.nn.embedding_lookup(self.features, sample_ids)
        #    # 输入是h_i+o_{i-1}+c_i
        #    aa=encoder_outputs[100]
        #    print('##########################')
        #    print(time)
        #    print(aa)
        #    print('##########################')
        #    #next_input2 = tf.concat((pred_feature, encoder_outputs[time]), 0)
        #    next_input = tf.concat((pred_feature, encoder_outputs[time]), 1)
        #    elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
        #    all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
        #    next_inputs = tf.cond(all_finished, lambda: pad_step_feature, lambda: next_input)
        #    next_state = state
        #    return elements_finished, next_inputs, next_state

        #my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        #def decode(helper, scope, reuse=None):
        #    with tf.variable_scope(scope, reuse=reuse):
        #        memory = tf.transpose(encoder_outputs, [1, 0, 2])
        #        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        #            num_units=self.hidden_size, memory=memory,
        #            memory_sequence_length=self.encoder_sequence_length)
        #        cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
        #        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        #            cell, attention_mechanism, attention_layer_size=self.hidden_size)
        #        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
        #            attn_cell, self.label_size, reuse=reuse
        #        )
        #        decoder = tf.contrib.seq2seq.BasicDecoder(
        #            cell=out_cell, helper=helper,
        #            initial_state=out_cell.zero_state(
        #                dtype=tf.float32, batch_size=batch_size))
        #        # initial_state=encoder_final_state)
        #        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        #            decoder=decoder, output_time_major=True,
        #            impute_finished=True
        #        )
        #        return final_outputs

        #outputs = decode(my_helper, 'decode')
        #self.decoder_prediction = tf.transpose(outputs.rnn_output, [1, 0,2])
        tf.add_to_collection('decoder_prediction', self.decoder_prediction)
       

        #
        mask=tf.to_float(tf.reshape(tf.sequence_mask(self.encoder_sequence_length,self.max_time),[self.batch_size*self.max_time,1]))
       
        self.loss=tf.losses.mean_squared_error(
                    labels=tf.reshape(self.decoder_targets,[self.batch_size*self.max_time,self.label_size]),
                    predictions=self.decoder_prediction,
                    
                    weights=mask
                    )
        tf.summary.scalar('loss', self.loss)
        tf.add_to_collection('loss', self.loss)
        
        #self.loss=tf.div(losses,tf.cast(tf.reduce_sum(self.encoder_sequence_length),tf.float32)) 

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

class seq2seq2(object):
    def __init__(self, args):
        self.feature_size = args.feature_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.label_size=args.label_size
        self.learning_rate=args.learning_rate
        self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=args.batch_size
        #self.batch_size=16
        #self.encoder_inputs_index = tf.placeholder(tf.int32, [None, None],name='encoder_inputs_index')
        #self.encoder_sequence_length = tf.placeholder(tf.int32, [None],name='encoder_sequence_length')
        #self.decoder_targets = tf.placeholder(tf.float32, [None, None, self.label_size],name='decoder_targets')
        self.seq2seq_model()

    def seq2seq_model(self):
        self.encoder_input = [tf.placeholder(tf.float32, shape=[self.batch_size,self.feature_size], name="encoder_input_{}".format(t)) for t in tf.range(self.max_time)]
        self.decoder_targets = [tf.placeholder(tf.float32, [self.batch_size,self.label_size],name='decoder_targets')]

       
        Ws_in = self._weight_variable([self.feature_size, self.hidden_size],name='wsin')
        bs_in = self._bias_variable([self.hidden_size,],name='bsin')
        scale_factor = tf.Variable(1.0, name="ScaleFactor")
        #encoder_2d_input=tf.reshape(self.encoder_input,[-1,self.feature_size])
        #net_input =tf.matmul(encoder_2d_input, Ws_in) + bs_in
        #encoder_3d_input=tf.reshape(net_input,[self.batch_size,self.max_time,self.hidden_size])
        cell_encoder_input =[scale_factor*(tf.matmul(i, Ws_in) + bs_in) for i in self.encoder_input]

        cell_decoder_input = tf.zeros_like(cell_encoder_input[0], dtype=np.float32, name="GO") + cell_encoder_input[:-1]
        #decoder_3d_input = tf.reshape(decoder_2d_input,[self.batch_size,self.max_time,self.feature_size])
        #cell_decoder_input =tf.unstack(decoder_3d_input,axis=1)
        
        
        cells = []
        for i in range(self.num_layers):
            with tf.variable_scope('Generator.LSTM_{}'.format(i)):
                cells.append(rnn_cell.DropoutWrapper(cell=rnn_cell.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True), input_keep_prob=1.0, output_keep_prob=self.output_keep_prob))
        cell = rnn_cell.MultiRNNCell(cells)

        decoder_output, decoder_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
            cell_encoder_input,
            cell_decoder_input,
            cell
           
          
        )
        
        Ws_out = self._weight_variable([self.hidden_size, self.label_size],name='wsout')
        bs_out = self._bias_variable([self.label_size,],name='bsout')
        self.decoder_prediction =[scale_factor*(tf.matmul(j, Ws_out) + bs_out) for j in decoder_output]
        #self.decoder_prediction=tf.nn.relu(pred)
        tf.add_to_collection('pred_network', self.decoder_prediction)

       
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(self.decoder_targets, self.decoder_prediction):
                output_loss += self.seq2seq_loss(_y,_Y)
                
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if not ("ScaleFactor" in tf_var.name):
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
                    
            self.loss = output_loss + lambda_l2_reg * reg_loss
       
        #self.loss=self.seq2seq_loss(self.decoder_targets,self.decoder_prediction)


        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.5,beta2=0.9).minimize(self.loss)

    def seq2seq_loss(self,real,fake):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(fake, [-1], name='reshape_pred')],
            [tf.reshape(real, [-1], name='reshape_target')],
            [tf.ones([self.batch_size*self.max_time*self.label_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
        )
        _loss = tf.div(
            tf.reduce_sum(losses),
            tf.cast(self.batch_size*self.max_time*self.label_size,tf.float32),
            )
        tf.summary.scalar('seq2seq_loss',_loss)
        tf.add_to_collection('loss', _loss)
        return _loss
    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))
    def _weight_variable(self, shape, name='weights'):
        #initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape,  name=name)

    def _bias_variable(self, shape, name='biases'):
        #initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape)


class seq_with_ScheduledOutputTrainingHelper(object):
    def __init__(self, args):
        self.input_dim = args.input_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.output_dim=args.output_dim
        self.learning_rate=args.learning_rate
        self.phase = tf.placeholder(tf.bool) # if True: training / if False: inference
        #self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
        self.max_time=20
        #self.batch_size=50
        self.sequence_length = tf.placeholder(tf.int32, [None],name='sequence_length')
        self.encoder_inputs = tf.placeholder(tf.float32, shape=[None,None,self.input_dim], name="encoder_inputs") 
        self.decoder_targets = tf.placeholder(tf.float32, [None,None,self.output_dim],name='decoder_targets')
        self.seq2seq_model()

    def encoder(self):
        with tf.variable_scope("encoder") as encoder_scope:
            encoder_w_in = self._weight_variable([self.input_dim, self.hidden_size],name='encoder_w_in')
            encoder_b_in = self._bias_variable([self.hidden_size,],name='encoder_b_in')
            encoder_inputs_2d=tf.reshape(self.encoder_inputs,[self.batch_size*self.max_time,self.input_dim])
            encoder_cell_inputs=tf.nn.relu(tf.add(tf.matmul(encoder_inputs_2d,encoder_w_in),encoder_b_in))
            encoder_cell_inputs_3d=tf.reshape(encoder_cell_inputs,[self.batch_size,self.max_time,self.hidden_size])

            
            encoder_fw_cells = []
            encoder_bw_cells = []
            for i in range(self.num_layers):
                with tf.variable_scope('encoder_lstm_{}'.format(i)):
                    encoder_fw_cells.append(rnn_cell.DropoutWrapper(cell=rnn_cell.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True), input_keep_prob=1.0, output_keep_prob=self.output_keep_prob))
                    encoder_bw_cells.append(rnn_cell.DropoutWrapper(cell=rnn_cell.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True), input_keep_prob=1.0, output_keep_prob=self.output_keep_prob))
            encoder_muti_fw_cell = rnn_cell.MultiRNNCell(encoder_fw_cells)
            encoder_muti_bw_cell = rnn_cell.MultiRNNCell(encoder_bw_cells)

            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_muti_fw_cell,
                                                cell_bw=encoder_muti_bw_cell,
                                                inputs=encoder_cell_inputs_3d,
                                                #sequence_length=self.sequence_length,
                                                dtype=tf.float32, time_major=False)

            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            #encoder_final_state_c = tf.concat(
            #    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

            #encoder_final_state_h = tf.concat(
            #    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

            #encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
            #    c=encoder_final_state_c,
            #    h=encoder_final_state_h
            #)
            return encoder_outputs
    def decoder(self,encoder_outputs):
        with tf.variable_scope("decoder") as decoder_scope:
            batch_size,max_time,_=tf.unstack(tf.shape(encoder_outputs))
            attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.hidden_size, 
                memory=encoder_outputs)
                #memory_sequence_length=self.sequence_length)
            #attn_initial_cell_state= (LSTMStateTuple(decoder_initial_state, decoder_initial_state))
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(2*self.hidden_size)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell,
                                    attention_mechanism=attention_mechanism, 
                                    #initial_cell_state =attn_initial_cell_state,
                                    alignment_history = True) 

            output_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell,self.output_dim)
            
            #sampling_prob = tf.cond(phase,
            #                        #training
            #                        lambda :tf.constant(1.0) - tf.train.inverse_time_decay
            #                        (learning_rate=1.0,
            #                         global_step=self.global_step,
            #                         decay_steps=1000,
            #                         decay_rate=0.9),
            #                        #inference
            #                        lambda : tf.constant(1.0))
            decoder_input = tf.zeros_like(self.decoder_targets,dtype=tf.float32,name='decoder_inputs')
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(decoder_input, 
                                                   sequence_length=self.sequence_length, 
                                                   sampling_probability=tf.constant(1.0))    
            
            my_decoder = tf.contrib.seq2seq.BasicDecoder(cell=output_cell, 
                                      helper=helper, 
                                      initial_state=output_cell.zero_state(
                                          batch_size=batch_size, dtype=tf.float32)
                                      )
            
            final_outputs, final_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(my_decoder, impute_finished = True)
                
            return final_outputs.rnn_output
    def seq2seq_model(self):
        
        encoder_outputs=self.encoder()
        decoder_outputs=self.decoder(encoder_outputs)
        self.decoder_prediction =tf.reshape(decoder_outputs,[-1,self.output_dim])
        tf.add_to_collection('pred_network', self.decoder_prediction)

       
        with tf.variable_scope('loss'):
            #mask=tf.to_float(tf.reshape(tf.sequence_mask(self.sequence_length,self.max_time),[self.batch_size*self.max_time,1]))
            
            labels=tf.reshape(self.decoder_targets,[self.batch_size*self.max_time,self.output_dim])
            loss_b=tf.losses.mean_squared_error(tf.reshape(labels[:,0],[-1,1]),tf.reshape(self.decoder_prediction[:,0],[-1,1]))
            loss_y=tf.losses.mean_squared_error(tf.reshape(labels[:,1],[-1,1]),tf.reshape(self.decoder_prediction[:,1],[-1,1]))
            #self.loss=loss_b+loss_y+tf.add_n(tf.get_collection("regular_loss"))
            self.loss=(loss_b+loss_y)/2
            tf.summary.scalar('loss', self.loss)
            tf.add_to_collection('loss', self.loss)
        with tf.variable_scope('tain_op'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.5,beta2=0.9).minimize(self.loss)
   
    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))
    def _weight_variable(self, shape, name='weights'):
        #initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        w=tf.get_variable(shape=shape,  name=name)
        #tf.add_to_collection("regular_loss",tf.contrib.layers.l2_regularizer(0.5)(w))
        return w

    def _bias_variable(self, shape, name='biases'):
        #initializer = tf.constant_initializer(0.1)
        b=tf.get_variable(name=name, shape=shape)
        #tf.add_to_collection("regular_loss",tf.contrib.layers.l2_regularizer(0.5)(b))
        return b

        