package k.kapi;

import java.util.ArrayList;

public class Main {


    static int in_size = 12;
    static int[] hiddens =  new int[]{10};
    static int out_size = 5;

    static int hm_epochs = 20;
    static float learning_rate = .1f;

    static int hidden_size = 10;

    static String activation_fn = "sigm";



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

        test_model();

    }

    static void test_model() {

        ArrayList<K_Layer> model = K_NN.FeedForward(in_size, new int[]{hidden_size}, out_size, activation_fn);

        K_Tensor t_out = K_NN.propogate(model, t_in);

        System.out.println(t_out);

    }

    static void test_layer() {

        K_Layer layer1 = new K_Layer(in_size, hidden_size, "sigm");
        K_Layer layer2 = new K_Layer(hidden_size, out_size, "sigm");

        K_Tensor out1 = K_NN.propogate(layer1, t_in);
        K_Tensor out2 = K_NN.propogate(layer2, out1);

        System.out.println(out2);

    }

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
