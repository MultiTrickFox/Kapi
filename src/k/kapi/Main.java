package k.kapi;

import java.util.ArrayList;
import java.util.List;

public class Main {


    static int in_size = 12;
    static int[] hiddens =  new int[]{20};
    static int out_size = 12;

    static int hm_epochs = 20;
    static float learning_rate = .1f;

    static int batch_size = 10;

    static String activation_fn = "sigm";

    static int hm_data = 100;
    static int seq_len = 20;



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



    public static void main(String[] args) {

        test_training_loop();

    }

    static void test_training_loop() {

        ArrayList<K_Tensor[]> dataset = K_Trainer.create_fake_data(in_size, out_size, hm_data, seq_len);

        ArrayList<LayerLSTM> model = K_Api.LSTM(in_size, hiddens, out_size);

        float ep_loss = 0;

        for (int i = 0; i < hm_epochs; i++) {

            ep_loss = 0;

            for (K_Tensor[][] batch : K_Trainer.batchify(dataset, batch_size))

                ep_loss += K_Trainer.batch_train(model, batch, learning_rate);

            System.out.println("Epoch " + i + " Loss " + ep_loss);

        }

    }

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

//    static void test_feedforw() {
//
//        List<LayerDense> model = K_Api.FeedForward(in_size, new int[]{hidden_size}, out_size, activation_fn);
//
//        K_Tensor t_out = K_Api.propogate(model, t_in);
//
//        System.out.println(t_out);
//
//    }

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
        K_Tensor.fill_grads(node_out);
        System.out.println("node out " + t_in.grad[0][0] + " " + t_w1.grad[0][0]);
        K_Tensor.empty_grads();


        K_Tensor node_out_gated = K_Tensor.sigm(node_out);
        K_Tensor.fill_grads(node_out_gated);
        System.out.println("node out gated " + t_in.grad[0][0] + " " + t_w1.grad[0][0]);
        K_Tensor.empty_grads();
//
//
        K_Tensor node_out2 = K_Tensor.matmul(node_out_gated, t_w2);
        K_Tensor.fill_grads(node_out2);
        System.out.println("node out 2 " + t_in.grad[0][0] + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
        K_Tensor.empty_grads();
//
//
        K_Tensor node_loss = K_Tensor.mean_square(target, node_out2);
        float loss = K_Tensor.fill_grads(node_loss);
        System.out.println("node loss " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
        K_Tensor.empty_grads();

        System.out.println("Loss: " + loss + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);


//        // update weights.
        t_w1.matrix = K_Math.sub(t_w1.matrix, K_Math.mul_scalar(t_w1.grad, 0.01f));
        t_w2.matrix = K_Math.sub(t_w2.matrix, K_Math.mul_scalar(t_w2.grad, 0.01f));
        K_Tensor.empty_grads();

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
            K_Tensor.empty_grads();

        }

    }


}