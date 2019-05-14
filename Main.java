import java.util.ArrayList;

public class Main {


    static int in_size = 12;
    static int[] hiddens =  new int[]{10};
    static int out_size = 5;

    static int hm_data = 10;
    static int hm_timesteps = 5;


    public static void main(String[] args) {


        System.out.println("Hello World!");


        K_base kb = new K_base();
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

        ArrayList<Double[][]>[] responses = K_api.batch_response(gru, data);

        System.out.println("" + responses[0].get(0)[0]);



    }
}
