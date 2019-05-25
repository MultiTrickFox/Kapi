public class Main {


    static int in_size = 12;
    static int[] hiddens =  new int[]{10};
    static int out_size = 5;


    static int hm_data = 10;
    static int hm_timesteps = 5;


    public static void main(String[] args) {

        // here lies the testing grounds..

        Double[][] in = new Double[1][2];
        Double[][] w1 = new Double[2][3];
        Double[][] w2 = new Double[3][1];
        Double[][] l = new Double[1][1];

        in[0][0] = 0.2;
        in[0][1] = 0.3;

        w1[0][0] = 0.3;
        w1[0][1] = 0.4;
        w1[0][2] = 0.5;
        w1[1][0] = 0.6;
        w1[1][1] = 0.1;
        w1[1][2] = 0.2;

        w2[0][0] = 0.7;
        w2[1][0] = 0.1;
        w2[2][0] = -0.2;

        l[0][0] = 0.1;

        // K_tensor t_in = new K_tensor(in); t_in.requires_grad=false;
        Double[][] t_in = l;
        K_tensor t_w1 = new K_tensor(w1); t_w1.requires_grad=true;
        K_tensor t_w2 = new K_tensor(w2); t_w2.requires_grad=true;


        K_tensor node_out = K_tensor.matmul(t_in, t_w1);
        K_tensor node_out_gated = K_tensor.sigm(node_out);

        K_tensor node_out2 = K_tensor.matmul(node_out_gated, t_w2);

        K_tensor node_loss = K_tensor.mean_square(l, node_out2);


        double loss = K_tensor.fill_grads(node_loss);

        System.out.println("Loss: " + loss + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);

        t_w1.matrix = K_math.sub(t_w1.matrix, K_math.mul_scalar(t_w1.grad, 0.01));
        t_w2.matrix = K_math.sub(t_w2.matrix, K_math.mul_scalar(t_w2.grad, 0.01));

        K_tensor.empty_grads();



        // repeat.
        node_out = K_tensor.matmul(t_in, t_w1);
        node_out_gated = K_tensor.sigm(node_out);

        node_out2 = K_tensor.matmul(node_out_gated, t_w2);

        node_loss = K_tensor.mean_square(l, node_out2);


        loss = K_tensor.fill_grads(node_loss);

        System.out.println("Loss: " + loss + " " + t_w1.grad[0][0] + " " + t_w2.grad[0][0]);

        K_tensor.empty_grads();



//        K_tensor tensor_x = K_tensor.constants(in_size, 3, 2.0);
//        K_tensor tensor_w = K_tensor.constants(3, out_size, 3.0);
//
//        K_tensor tensor_lbl = K_tensor.constants(1, out_size, 7.0);
//
//        double loss = K_tensor.fill_grads(K_tensor.mean_square(tensor_lbl, K_tensor.matmul(tensor_x, tensor_w)));
//
//        System.out.println("Loss value: " + loss);
//        System.out.println("Grad w[0][0] value: " + tensor_w.grad[0][0]);
//        System.out.println("Grad w[0][1] value: " + tensor_w.grad[0][1]);
//
//        tensor_w.matrix = K_math.sub(tensor_w.matrix, K_math.mul_scalar(tensor_w.grad, 0.001));
//
//        K_tensor.empty_grads();
//
//
//        loss = K_tensor.fill_grads(K_tensor.mean_square(tensor_lbl, K_tensor.matmul(tensor_x, tensor_w)));
//
//        System.out.println("Loss value: " + loss);
//        System.out.println("Grad w[0][0] value: " + tensor_w.grad[0][0]);
//        System.out.println("Grad w[0][1] value: " + tensor_w.grad[0][1]);
//
//        K_tensor.empty_grads();



    }





}
