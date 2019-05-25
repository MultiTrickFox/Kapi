import java.util.ArrayList;

public class Main {


    static int in_size = 12;
    static int[] hiddens =  new int[]{10};
    static int out_size = 5;


    static int hm_data = 10;
    static int hm_timesteps = 5;


    public static void main(String[] args) {


        System.out.println("Hello World!");


        K_math kb = new K_math();
        K_api kapi = new K_api();

        GRU gru = (GRU) kapi.make_model("gru", in_size, hiddens, out_size);

        ArrayList<ArrayList<Double[][]>> data = new ArrayList<>();


        for (int i = 0; i < hm_data; i++) {

            ArrayList<Double[][]> sequence = new ArrayList<>();

            for (int k = 0; k < hm_timesteps; k++)

                sequence.add(kb.randn(1,in_size));

            data.add(sequence);

        }


//        ArrayList<Double[][]> response = gru.respond_to(data.get(0));
//
//        System.out.println(response.get(0)[0][1]);

        // ArrayList<Double[][]>[] responses = K_api.batch_response(gru, data);

        // System.out.println("" + responses[0].get(0)[0]);


        Double[][] matrix1 = new Double[1][2];
        Double[][] matrix2 = new Double[2][3];
        Double[][] matrix3 = new Double[3][3];

        Double[][] matrix4 = new Double[1][2];
        Double[][] matrix5 = new Double[1][2];

        K_tensor label = K_tensor.randn(1,3);

        for (int i = 0; i < matrix1.length; i++)
            for (int j = 0; j < matrix1[0].length; j++)
                    matrix1[i][j] = 0.02;

        for (int i = 0; i < matrix2.length; i++)
            for (int j = 0; j < matrix2[0].length; j++)
                matrix2[i][j] = 0.21;

        for (int i = 0; i < matrix3.length; i++)
            for (int j = 0; j < matrix3[0].length; j++)
                matrix3[i][j] = 4.0;

        for (int i = 0; i < matrix4.length; i++)
            for (int j = 0; j < matrix4[0].length; j++)
                matrix4[i][j] = 0.001;

        for (int i = 0; i < matrix5.length; i++)
            for (int j = 0; j < matrix5[0].length; j++)
                matrix5[i][j] = 0.4;

        K_tensor tensor1 = new K_tensor(matrix1);
        K_tensor tensor2 = new K_tensor(matrix2);
        K_tensor tensor3 = new K_tensor(matrix3); //K_tensor.randn(1,3);
        K_tensor tensor4 = new K_tensor(matrix4); //K_tensor.randn(1,3);
        K_tensor tensor5 = new K_tensor(matrix5); //K_tensor.randn(1,3);

//        K_tensor result = K_tensor.matmul(tensor1, tensor2);

        K_tensor mul = K_tensor.mul(tensor1, tensor4);

        K_tensor result = K_tensor.cross_entropy(tensor5, K_tensor.softmax(mul));
//
//        K_tensor.make_grads(result);
//        System.out.println("" + tensor1.grads[0][0] + " " + tensor2.grads[0][1]);
//        K_tensor.erase_grads();

        // K_tensor result2 = K_tensor.cross_entropy(label, result);

        K_tensor.make_grads(result);

        //System.out.println("" + mul.grads[0][0]);

        // K_tensor.make_grads(result2);
        System.out.println("" + tensor1.grads[0][0] + " " + tensor4.grads[0][0]);
        K_tensor.erase_grads();

//        K_tensor result_sigm = K_tensor.sigm(result);
//
//        K_tensor result2 = K_tensor.matmul(result_sigm, tensor3);
//
//        K_tensor soft = K_tensor.softmax(result2);
//
//        K_tensor loss = K_tensor.cross_entropy(target, soft);
//
//        K_tensor.make_grads(loss);
//
//        System.out.println("" + tensor1.grads[0][0] + " " + tensor2.grads[0][0] + " " + tensor2.grads[0][0]);
//
//        K_tensor.erase_grads();
//




    }





}
