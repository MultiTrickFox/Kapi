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

        for (int i = 0; i < matrix1.length; i++)
            for (int j = 0; j < matrix1[0].length; j++)
                    matrix1[i][j] = 3.0;

        for (int i = 0; i < matrix2.length; i++)
            for (int j = 0; j < matrix2[0].length; j++)
                matrix2[i][j] = 3.0;

        K_tensor tensor1 = new K_tensor(matrix1);
        K_tensor tensor2 = new K_tensor(matrix2);

        K_tensor result = K_tensor.matmul(tensor1, tensor2);

        System.out.println("" + K_tensor.size(result)[0] + " " + K_tensor.size(result)[1]);

        K_tensor.fill(result);

        System.out.println("" + tensor1.grads[0][1] + " " + tensor2.grads[0][1]);

        K_tensor.empty();

        System.out.println("" + tensor1.grads[0][1] + " " + tensor2.grads[0][1]);

    }





}
