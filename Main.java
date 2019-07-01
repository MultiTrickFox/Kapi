public class Main {


    static int in_size = 12;
    static int[] hiddens =  new int[]{10};
    static int out_size = 5;


    static int hm_data = 10;
    static int hm_timesteps = 5;


    public static void main(String[] args) {

        // here lies the testing grounds..


        Double[][] in = new Double[3][3];
        Double[][] w1 = new Double[3][3];
        Double[][] w2 = new Double[3][3];
        Double[][] l = new Double[1][1];

        in[0][0] = 0.2;
        in[0][1] = 0.3;
        in[0][2] = 0.1;
        in[1][0] = 0.1;
        in[1][1] = 0.4;
        in[1][2] = 0.7;
        in[2][0] = 0.1;
        in[2][1] = 0.6;
        in[2][2] = 0.4;

        w1[0][0] = 0.1;
        w1[0][1] = 0.6;
        w1[0][2] = 0.2;
        w1[1][0] = 0.7;
        w1[1][1] = -0.2;
        w1[1][2] = -0.1;
        w1[2][0] = 0.2;
        w1[2][1] = 0.7;
        w1[2][2] = 0.1;

        w2[0][0] = 0.3;
        w2[0][1] = 0.1;
        w2[0][2] = 0.01;
        w2[1][0] = -0.1;
        w2[1][1] = -0.1;
        w2[1][2] = -0.2;
        w2[2][0] = -0.3;
        w2[2][1] = 0.1;
        w2[2][2] = 0.2;

        l = K_math.constants(3,3, 3);

        K_tensor t_in = new K_tensor(in); t_in.requires_grad=true;
        K_tensor t_w1 = new K_tensor(w1); t_w1.requires_grad=true;
        K_tensor t_w2 = new K_tensor(w2); t_w2.requires_grad=true;



//        K_tensor node_test1 = K_tensor.matmul(t_in, t_w1);
//        K_tensor.fill_grads(node_test1);
//        K_tensor.empty_grads();


//        Double[][] smt = new Double[1][2];
//        smt[0][0] = 0.2;
//        smt[0][1] = 0.3;
//        t_in = new K_tensor(smt);
//        t_in.requires_grad = true;
//
//        K_tensor node_test2 = K_tensor.softmax(t_in); // K_tensor.matmul(K_tensor.sigm(K_tensor.matmul(t_in, t_w1)), t_w2);
//
//
//        K_tensor.fill_grads(node_test2);
//        K_tensor.empty_grads();

//
//        node_test = K_tensor.matmul(K_tensor.matmul(t_in, t_w1), t_w2);
//        K_tensor.fill_grads(node_test);
//        System.out.println("node test 2:  " + t_in.matrix[0][0] + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
//        K_tensor.empty_grads();



        K_tensor node_out = K_tensor.matmul(t_in, t_w1);
        //K_tensor.fill_grads(node_out);
        //System.out.println("node out " + t_in.grad[0][0] + " " + t_w1.grad[0][0]);
        //K_tensor.empty_grads();


        K_tensor node_out_gated = K_tensor.sigm(node_out);
        //K_tensor.fill_grads(node_out_gated);
        //System.out.println("node out gated " + t_in.grad[0][0] + " " + t_w1.grad[0][0]);
        //K_tensor.empty_grads();


        K_tensor node_out2 = K_tensor.matmul(node_out_gated, t_w2);
        K_tensor.fill_grads(node_out2);
        //System.out.println("node out 2 " + t_in.grad[0][0] + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
        K_tensor.empty_grads();


        K_tensor node_loss = K_tensor.mean_square(l, node_out2);
        K_tensor.fill_grads(node_loss);
        System.out.println("node loss " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
        K_tensor.empty_grads();




//        // update weights.
//        double loss = K_tensor.fill_grads(node_loss);
//        System.out.println("Loss: " + loss + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
//        t_w1.matrix = K_math.sub(t_w1.matrix, K_math.mul_scalar(t_w1.grad, 0.01));
//        t_w2.matrix = K_math.sub(t_w2.matrix, K_math.mul_scalar(t_w2.grad, 0.01));
//        K_tensor.empty_grads();
//
//
//
//        // repeat.
//        node_out = K_tensor.matmul(t_in, t_w1);
//        node_out_gated = K_tensor.sigm(node_out);
//        node_out2 = K_tensor.matmul(node_out_gated, t_w2);
//        node_loss = K_tensor.mean_square(l, node_out2);
//
//
//        loss = K_tensor.fill_grads(node_loss);
//        System.out.println("Loss: " + loss + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);
//        K_tensor.empty_grads();

    }

}
