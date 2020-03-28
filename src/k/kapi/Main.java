package k.kapi;

//import RootBeer.rootbeer.runtime.Kernel; //
//import RootBeer.rootbeer.runtime.Rootbeer;


import java.util.ArrayList;
import java.util.List;




public class Main {


    static int in_size = 13;
    static int[] hiddens =  new int[]{3, 2};
    static int out_size = 13;

    static int hm_epochs = 20;
    static int test_per_epochs = 5;

    static float learning_rate = .1f;

    static int batch_size = 4;

    static String activation_fn = "sigm";

    static int hm_data = 10;
    static int seq_len = 10;

    static float train_ratio = .8f;
    static float test_ratio = .2f;



    public static void main(String[] args) {

        //test_generic_model();

        //test_training_loop();

        //test_trainer();

        //test_biggest_trainer();

        K_CL.init();

        //test_threading_gpu();

        test_custom();



        //test_model_combining();\

        //test_gpu_stuff();


//        Float[][] result = GPU_OPS.matmul(K_Math.zeros(2,3), K_Math.zeros(3,4));
//
//        System.out.println(result);



        System.out.println("End..");




    }

    static void test_threading_gpu() {

        ArrayList<ArrayList<Float[][]>> dataset = create_fake_data(13, 13, 4, 3);

        List<Object> model = K_Api.Generate_Generic_Model(

                new int[]{in_size,hiddens[0],hiddens[1],out_size},
                new String[]{"dense","lstm","dense"},
                "elu");

        //Object[] loss_grad = K_Api.loss_and_grad_from_datapoint(model, dataset.get(3));

        Object[] loss_grad = K_Api.loss_and_grad_from_batch(model, K_Utils.batchify(dataset,2).get(0));

    }


    static void test_custom() {

        ArrayList<ArrayList<ArrayList<Float[][]>>> dataset = create_fake_data2(13, 13, 10, 3);

//                K_Dlc.Encoder_Decoder encdec = new K_Dlc.Encoder_Decoder(
//                    new int[]{13,3,2,13},
//                    new String[]{"dense","lstm","dense"},
//                    "sigm");

        K_Dlc.Encoder_Decoder encdec = new K_Dlc.Encoder_Decoder(
                new int[]{ 13,      3,     2,      13},
                new String[]{"dense","lstm","dense"},
                new int[]{ 13,      3,     2,     2,     13},
                new String[]{"dense","lstm","lstm","dense"},
                "sigm");

        //Object[] results = K_Dlc.loss_and_grad_from_datapoint(encdec, dataset.get(0).get(0), dataset.get(0).get(1));

        //System.out.println(results);


        float loss = K_Dlc.train_on_batch(encdec, K_Utils.batchify2(dataset, 5).get(0), learning_rate);

        System.out.println(loss);

        loss = K_Dlc.train_on_batch(encdec, K_Utils.batchify2(K_Utils.shuffle2(dataset), 5).get(0), learning_rate);

        System.out.println(loss);

        //K_Dlc.loss_and_grad_from_datapoint(encdec, dataset.get(0).get(0), dataset.get(0).get(1));

    }

    static void test_model_combining() {

        ArrayList<ArrayList<Float[][]>> dataset = create_fake_data(13, 13, 4, 3);

        List<Object> model = K_Api.Generate_Generic_Model(

                new int[]{in_size,hiddens[0],hiddens[1],out_size},
                new String[]{"dense","lstm","dense"},
                "elu");

        List<Object> model2 = K_Api.Generate_Generic_Model(

                new int[]{in_size,hiddens[0],hiddens[1],out_size},
                new String[]{"dense","lstm","dense"},
                "elu");

        K_Layer.Dense model3 = new K_Layer.Dense(out_size, out_size, "sigm");

        K_Layer.LSTM model4 = new K_Layer.LSTM(out_size, out_size);

        List<Object> modelx = K_Utils.combine_networks(model, model2);
        List<Object> modely = K_Utils.combine_networks(model, model3);
        List<Object> modelz = K_Utils.combine_networks(model, model4);
        List<Object> modelq = K_Utils.combine_networks(model3, model4);


        K_Api.train_on_batch(modelx, dataset, learning_rate);
        K_Api.train_on_batch(modely, dataset, learning_rate);
        K_Api.train_on_batch(modelz, dataset, learning_rate);
        K_Api.train_on_batch(modelq, dataset, learning_rate);

    }

    static void test_biggest_trainer() {

        ArrayList<ArrayList<Float[][]>> dataset = create_fake_data(in_size, out_size, hm_data, seq_len);

        //ArrayList<K_Layer.LSTM> model = K_Model.LSTM(in_size, hiddens, out_size);

        List<Object> model = K_Api.Generate_Generic_Model(

                new int[]{in_size,hiddens[0],hiddens[1],out_size},
                new String[]{"dense","lstm","dense"},
                "elu");

//        ArrayList<K_Layer.LSTM> model = K_Model.LSTM(in_size, hiddens, out_size);


        K_Api.Train_On_Dataset(model, dataset, train_ratio, test_ratio, batch_size, learning_rate, hm_epochs, test_per_epochs);

    }

    static void test_generic_model() {

        List<Object> model = K_Api.Generate_Generic_Model(

                new int[]{in_size,hiddens[0],hiddens[1],out_size},
                new String[]{"dense","lstm","dense"},
                "elu");

        ArrayList<ArrayList<Float[][]>> dataset = create_fake_data(in_size, out_size, hm_data, seq_len);

        K_Api.Train_On_Dataset(model, dataset, batch_size, learning_rate, hm_epochs);

        //K_Api.loss_and_grad_from_datapoint(model, K_Util.shuffle(dataset).get(0));

    }

    static void test_trainer() {

        ArrayList<ArrayList<Float[][]>> dataset = create_fake_data(in_size, out_size, hm_data, seq_len);

        ArrayList<K_Layer.LSTM> model = K_Model.LSTM(in_size, hiddens, out_size);

        K_Api.Train_On_Dataset(model, dataset, batch_size, .001f, hm_epochs);

    }

    static void test_training_loop() {

        ArrayList<ArrayList<Float[][]>> dataset = create_fake_data(in_size, out_size, hm_data, seq_len);

        ArrayList<K_Layer.LSTM> model = K_Model.LSTM(in_size, hiddens, out_size);

        float ep_loss = 0;

        for (int i = 0; i < hm_epochs; i++) {

            ep_loss = 0;

            for (ArrayList<ArrayList<Float[][]>> batch : K_Utils.batchify(K_Utils.shuffle(dataset), batch_size))

                ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

            System.out.println("Epoch " + i + " Loss " + ep_loss);

        }

    }


    static Float[][] inp = new Float[3][3];
    static {
        inp[0][0] = 0.2f;
        inp[0][1] = 0.3f;
        inp[0][2] = 0.1f;
        inp[1][0] = 0.1f;
        inp[1][1] = 0.4f;
        inp[1][2] = 0.7f;
        inp[2][0] = 0.1f;
        inp[2][1] = 0.6f;
        inp[2][2] = 0.4f;
    }
    static K_Tensor t_in = new K_Tensor(inp);
    static Float[][] target = K_Math.constants(3,3, 3);
    static K_Tensor t_trg = new K_Tensor(target);

//    static void test_lstm() {
//
//        K_Tensor input = K_Tensor.randn(1, in_size);
//        K_Tensor label = K_Tensor.randn(1, out_size);
//
//        ArrayList<LayerLSTM> model = K_Api.LSTM(in_size, new int[]{hidden_size}, out_size);
//
//        K_Tensor t_out = K_Api.propogate(model, input);
//
//        System.out.println(t_out);
//
//    }

    static void test_feedforw() {

        List<Object> model = K_Model.FeedForward(in_size, new int[]{12}, out_size, activation_fn);

        K_Tensor[] t_out = K_Model.propogate(model, new K_Tensor[]{t_in});

        System.out.println(t_out[0]);

    }

//    static void test_layer() {
//
//        LayerDense layer1 = new LayerDense(in_size, hidden_size, "sigm");
//        LayerDense layer2 = new LayerDense(hidden_size, out_size, "sigm");
//
//        K_Tensor out1 = K_Api.propogate(layer1, t_in);
//        K_Tensor out2 = K_Api.propogate(layer2, out1);
//
//        System.out.println(out2);
//
//    }

    static void test_diff() {

        Float[][] w1 = new Float[3][3];
        Float[][] w2 = new Float[3][3];

        w1[0][0] = 0.1f;
        w1[0][1] = 0.6f;
        w1[0][2] = 0.2f;
        w1[1][0] = 0.7f;
        w1[1][1] = -0.2f;
        w1[1][2] = -0.1f;
        w1[2][0] = 0.2f;
        w1[2][1] = 0.7f;
        w1[2][2] = 0.1f;

        w2[0][0] = 0.3f;
        w2[0][1] = 0.1f;
        w2[0][2] = 0.01f;
        w2[1][0] = -0.1f;
        w2[1][1] = -0.1f;
        w2[1][2] = -0.2f;
        w2[2][0] = -0.3f;
        w2[2][1] = 0.1f;
        w2[2][2] = 0.2f;


        K_Tensor t_in = new K_Tensor(inp); t_in.requires_grad=true;
        K_Tensor t_w1 = new K_Tensor(w1); t_w1.requires_grad=true;
        K_Tensor t_w2 = new K_Tensor(w2); t_w2.requires_grad=true;


        K_Tensor node_out = K_Tensor.matmul(t_in, t_w1);
//        K_Tensor.fill_grads(node_out);
//        System.out.println("node out " + t_in.grad[0][0] + " " + t_w1.grad[0][0]);
//        K_Tensor.empty_grads();


        K_Tensor node_out_gated = K_Tensor.sigm(node_out);
//        K_Tensor.fill_grads(node_out_gated);
//        System.out.println("node out gated " + t_in.grad[0][0] + " " + t_w1.grad[0][0]);
//        K_Tensor.empty_grads();
//
//
        K_Tensor node_out2 = K_Tensor.matmul(node_out_gated, t_w2);
//        K_Tensor.fill_grads(node_out2);
//        System.out.println("node out 2 " + t_in.grad[0][0] + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
//        K_Tensor.empty_grads();
//
//
        K_Tensor node_loss = K_Tensor.mean_square(target, node_out2);
        float loss = K_Tensor.fill_grads(node_loss);
        System.out.println("node loss " + t_in.grad[0][0] + " " + t_w1.grad[0][0]);
//        K_Tensor.empty_grads();

        System.out.println("Loss: " + loss + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);


//        // update weights.
        t_w1.matrix = K_Math.sub(t_w1.matrix, K_Math.mul_scalar(t_w1.grad, 0.01f));
        t_w2.matrix = K_Math.sub(t_w2.matrix, K_Math.mul_scalar(t_w2.grad, 0.01f));
//        K_Tensor.empty_grads();

        for (int ep = 0; ep < hm_epochs; ep++) {

            // repeat.
            node_out = K_Tensor.matmul(t_in, t_w1);
            node_out_gated = K_Tensor.sigm(node_out);
            node_out2 = K_Tensor.matmul(node_out_gated, t_w2);
            node_loss = K_Tensor.mean_square(target, node_out2);

            loss = K_Tensor.fill_grads(node_loss);
            System.out.println("Ep " + ep + " Loss: " + loss + ", grads: " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
            t_w1.matrix = K_Math.sub(t_w1.matrix, K_Math.mul_scalar(t_w1.grad, 0.01f));
            t_w2.matrix = K_Math.sub(t_w2.matrix, K_Math.mul_scalar(t_w2.grad, 0.01f));
//            K_Tensor.empty_grads();

        }

    }


    static ArrayList<ArrayList<Float[][]>> create_fake_data(int in_size, int out_size, int hm_data, int max_length) {

        assert in_size == out_size;

        ArrayList<ArrayList<Float[][]>> dataset = new ArrayList<>();

        ArrayList<Float[][]> sequence;

        for (int i = 0; i < hm_data; i++) {

            sequence = new ArrayList<>();

            for (int t = 0; t < max_length; t++)

                sequence.add(K_Math.randn(1, in_size));

            dataset.add(sequence);

        }

        return dataset;

    }


    static ArrayList<ArrayList<ArrayList<Float[][]>>> create_fake_data2(int in_size, int out_size, int hm_data, int max_length) {

        assert in_size == out_size;

        ArrayList<ArrayList<ArrayList<Float[][]>>> dataset = new ArrayList<>();

        ArrayList<ArrayList<Float[][]>> datapoint;

        ArrayList<Float[][]> sequence_x;
        ArrayList<Float[][]> sequence_y;

        for (int i = 0; i < hm_data; i++) {

            datapoint = new ArrayList<>();

            sequence_x = new ArrayList<>();

            for (int t = 0; t < max_length; t++)

                sequence_x.add(K_Math.randn(1, in_size));

            datapoint.add(sequence_x);

            sequence_y = new ArrayList<>();

            for (int t = 0; t < max_length; t++)

                sequence_y.add(K_Math.randn(1, in_size));

            datapoint.add(sequence_y);

            dataset.add(datapoint);

        }

        return dataset;

    }



    static void test_gpu_stuff() {

        K_CL.init();

        System.out.println(K_CL.gpu_enabled);

        Float[][] marr = K_Math.randn(1_000,2_000);
        Float[][] marr2 = K_Math.randn(2_000,5_000);

        Float[][] res = K_CL.OPS.matmul(marr, marr2);

        System.out.println(res);

        Float[][] res_valid = K_Math.matmul(marr,marr2);

        if (K_Math.sum(res) == K_Math.sum(res_valid)) {

            System.out.println("Validation passed.");

        } else {
            System.out.println("Valid faild." + K_Math.sum(res) + " " + K_Math.sum(res_valid));
        }

    }



}




//// Rootbeer stuff
//
//
//
//class GPU_OPS {
//
//
//    static class Matmul_GPU implements Kernel {
//
//        Float[][] src1;
//        Float[][] src2;
//        Float[][] dest;
//        int i;
//        int j;
//
//        Matmul_GPU(Float[][] src1, Float[][] src2, Float[][] dest, int i, int j) {
//
//            this.src1 = src1;
//            this.src2 = src2;
//            this.dest = dest;
//            this.i = i;
//            this.j = j;
//
//        }
//
//        @Override
//        public void gpuMethod() {
//
//            this.dest[i][j] = 0.0f;
//            for (int k = 0; k < this.src1[0].length; k++)
//                this.dest[i][j] += this.src1[i][k] * this.src2[k][j];
//
//        }
//
//    }
//
//    static Float[][] matmul(Float[][] src1, Float[][] src2) {
//
//        Float[][] dest = new Float[src1.length][src2[0].length];
//
//        List<Kernel> tasks = new ArrayList<>();
//
//        for (int i = 0; i < src1.length; i++)
//
//            for (int j = 0; j < src2[0].length; j++)
//
//                tasks.add(new Matmul_GPU(src1, src2, dest, i, j));
//
//        Rootbeer rootbeer = new Rootbeer();
//
//        rootbeer.runAll(tasks);
//
//        return dest;
//
//    }
//
//}