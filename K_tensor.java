import java.util.ArrayList;


class K_tensor {


    Double[][] matrix;
    Double[][] grad;

    ArrayList<K_tensor> parents;
    ArrayList<K_tensor> childs;

    ArrayList<Double[][]> parent_grads;

    static ArrayList<K_tensor> graph = new ArrayList<>();

    boolean requires_grad = false;

    String type = "tensor";


    // constructor

    K_tensor(Double[][] matrix) {

        this.matrix = matrix;

        int[] size = K_math.size(matrix);
        this.grad = K_math.zeros(size[0], size[1]);

        this.parents = new ArrayList<>();
        this.childs = new ArrayList<>();
        this.parent_grads = new ArrayList<>();

        graph.add(this);

    }

    K_tensor() { } static K_tensor instance = new K_tensor();


    // matrix initializers

    static K_tensor zeros(int hm_rows, int hm_cols) {

        K_tensor tensor = new K_tensor(K_math.zeros(hm_rows, hm_cols));

        tensor.requires_grad = true;

        return tensor;

    }

    static K_tensor ones(int hm_rows, int hm_cols) {

        K_tensor tensor = new K_tensor(K_math.ones(hm_rows, hm_cols));

        tensor.requires_grad = true;

        return tensor;

    }

    static K_tensor randn(int hm_rows, int hm_cols) {

        K_tensor tensor = new K_tensor(K_math.randn(hm_rows, hm_cols));

        tensor.requires_grad = true;

        return tensor;

    }

    static K_tensor identity(int hm_rows, int hm_cols) {

        K_tensor tensor = new K_tensor(K_math.identity(hm_rows, hm_cols));

        tensor.requires_grad = true;

        return tensor;

    }

    static K_tensor constants(int hm_rows, int hm_cols, double val) {

        K_tensor tensor = new K_tensor(K_math.constants(hm_rows, hm_cols, val));

        tensor.requires_grad = true;

        return tensor;

    }


    // matrix operations  // todo : add inplace's

    static K_tensor add(K_tensor t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.add(t1.matrix, t2.matrix));

        define_as_child(tensor, t1);
        define_as_child(tensor, t2);

        int[] size_p1 = K_math.size(t1.matrix);
        int[] size_p2 = K_math.size(t2.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));
        tensor.parent_grads.add(K_math.ones(size_p2[0],size_p2[1]));

        return tensor;

    }

    static K_tensor add(K_tensor t1, Double[][] t2) {

        K_tensor tensor = new K_tensor(K_math.add(t1.matrix, t2));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    } static K_tensor add(Double[][] t1, K_tensor t2) { return add(t2, t1); }

    static K_tensor sub(K_tensor t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.sub(t1.matrix, t2.matrix));

        define_as_child(tensor, t1);
        define_as_child(tensor, t2);

        int[] size_p1 = K_math.size(t1.matrix);
        int[] size_p2 = K_math.size(t2.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));
        tensor.parent_grads.add(K_math.constants(size_p2[0],size_p2[1], -1));

        return tensor;

    }

    static K_tensor sub(K_tensor t1, Double[][] t2) {

        K_tensor tensor = new K_tensor(K_math.sub(t1.matrix, t2));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    }

    static K_tensor sub(Double[][] t2, K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.sub(t1.matrix, t2));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    }

    static K_tensor mul(K_tensor t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.mul(t1.matrix, t2.matrix));

        define_as_child(tensor, t1);
        define_as_child(tensor, t2);

        tensor.parent_grads.add(t2.matrix);
        tensor.parent_grads.add(t1.matrix);

        return tensor;

    }

    static K_tensor mul(K_tensor t1, Double[][] t2) {

        K_tensor tensor = new K_tensor(K_math.mul(t1.matrix, t2));

        define_as_child(tensor, t1);

        tensor.parent_grads.add(t2);

        return tensor;

    } static K_tensor mul(Double[][] t1, K_tensor t2) { return mul(t2, t1); }

    static K_tensor div(K_tensor t1, K_tensor t2) {

        return mul(t1, pow(t2, -1));

    }

    static K_tensor div(K_tensor t1, Double[][] t2) {

        return mul(t1, K_math.pow(t2, -1));

    }

    static K_tensor div(Double[][] t2, K_tensor t1) {

        return mul(t2, pow(t1, -1));

    }

    static K_tensor matmul(K_tensor t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.matmul(t1.matrix, t2.matrix));
        tensor.type = "matmul";

        define_as_child(tensor, t1);
        define_as_child(tensor, t2);

        tensor.parent_grads.add(t2.matrix);
        tensor.parent_grads.add(t1.matrix);

        return tensor;

    }

    static K_tensor matmul(K_tensor t1, Double[][] t2) {

        K_tensor tensor = new K_tensor(K_math.matmul(t1.matrix, t2));
        tensor.type = "matmul";

        define_as_child(tensor, t1);

        tensor.parent_grads.add(t2);

        return tensor;

    }

    static K_tensor matmul(Double[][] t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.matmul(t1, t2.matrix));
        tensor.type = "matmul";

        define_as_child(tensor, t2);

        tensor.parent_grads.add(t1);

        return tensor;

    }


    // scalar operations

    static K_tensor add(K_tensor t1, Double s) {

        K_tensor tensor = new K_tensor(K_math.add_scalar(t1.matrix, s));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    } static K_tensor add(Double s, K_tensor t1) { return add(t1, s); }

    static K_tensor sub(K_tensor t1, Double s) {

        return add(t1, -s);

    }

    static K_tensor sub(Double s, K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.sub_scalar(s, t1.matrix));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.constants(size_p1[0],size_p1[1], -1));

        return tensor;

    }

    static K_tensor mul(K_tensor t1, Double s) {

        K_tensor tensor = new K_tensor(K_math.mul_scalar(t1.matrix, s));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.constants(size_p1[0],size_p1[1], s));

        return tensor;

    } static K_tensor mul(Double s, K_tensor t1) { return mul(t1, s); }

    static K_tensor div(Double s, K_tensor t1) {

        return mul(s, pow(t1, -1));

    } static K_tensor div(K_tensor t1, Double s) { return mul(t1, 1/s); }


    // graph helpers

//    private static void define_as_same(K_tensor t1, K_tensor t2) {
//
//        t1.childs.addAll(t2.childs);
//        t1.parents.addAll(t2.parents);
//
//        for (K_tensor child : t2.childs)
//            child.parents.add(t1);
//
//        for (K_tensor parent : t2.parents)
//            parent.childs.add(t1);
//
//        t1.parent_grads = t2.parent_grads;
//        t1.grad = t2.grad; // this line wouldn't be correct.
//
//    }

    private static void define_as_child(K_tensor t1, K_tensor t2) {

        t1.parents.add(t2);
        t2.childs.add(t1);

    }

    static double fill_grads(K_tensor t1) {

        Double[][] incoming = K_math.ones(K_math.size(t1.matrix, 0), K_math.size(t1.matrix, 1));

        fill_grads(t1, incoming);

        return K_math.sum(t1.matrix);

    }

    private static void fill_grads(K_tensor t1, Double[][] incoming) {

        if (t1.requires_grad)
            t1.grad = K_math.add(t1.grad, incoming);

        if (t1.type.equals("matmul")) {

            int parent_ctr = -1;
            for (K_tensor parent : t1.parents) {
                parent_ctr++;

                fill_grads(parent,matmul_backwards(t1.parent_grads.get(parent_ctr), incoming, parent_ctr==0 ? "A" : "B"));

            }

        } else {

            int parent_ctr = -1;
            for (K_tensor parent : t1.parents) {
                parent_ctr++;

                fill_grads(parent, K_math.mul(t1.parent_grads.get(parent_ctr), incoming));

            }

        }

    }

    static void empty_grads() {

        for (K_tensor tensor : graph)

            tensor.grad = K_math.zeros(size(tensor, 0), size(tensor, 1));

        graph = new ArrayList<>();

    }

    private static Double[][] matmul_backwards(Double[][] parent_grad, Double[][] incoming_grad, String for_matrix) {

        int grad_rows = parent_grad.length;
        int grad_cols = parent_grad[0].length;
        int incoming_rows = incoming_grad.length;
        int incoming_cols = incoming_grad[0].length;

        if (for_matrix.equals("A")) {

            int outgoing_rows = incoming_rows;
            int outgoing_cols = grad_rows;
            Double[][] out = new Double[outgoing_rows][outgoing_cols];

            Double[] incoming_row, parent_row, outgoing_row;

            for (int i = 0; i < outgoing_rows; i++) {

                outgoing_row = out[i];
                incoming_row = incoming_grad[i];

                for (int j = 0; j < outgoing_cols; j++) {

                    parent_row = parent_grad[j];

                    outgoing_row[j] = K_math.vector_sum(K_math.vector_mul(incoming_row, parent_row));

                }

            }

            return out;

        }

        if (for_matrix.equals("B")) {

            int outgoing_rows = grad_cols;
            int outgoing_cols = incoming_cols;
            Double[][] out = new Double[outgoing_rows][outgoing_cols];

            Double[] incoming_row, parent_row, outgoing_row;

            incoming_grad = K_math.transpose(incoming_grad);
            parent_grad = K_math.transpose(parent_grad);

            for (int i = 0; i < outgoing_rows; i++) {

                outgoing_row = out[i];
                parent_row = parent_grad[i];

                for (int j = 0; j < outgoing_cols; j++) {

                    incoming_row = incoming_grad[i];

                    outgoing_row[j] = K_math.vector_sum(K_math.vector_mul(incoming_row, parent_row));

                }

            }

            return out;

        }

        return null;

    }


    // tensor helpers

    static int[] size(K_tensor t1) {

        return K_math.size(t1.matrix);

    }

    static int size(K_tensor t1, int dim) {

        return K_math.size(t1.matrix, dim);

    }

//    static K_tensor resize(K_tensor t1, int[] sizes) {
//
//        K_tensor tensor = new K_tensor(K_math.resize(t1.matrix, sizes));
//
//        define_as_same(tensor, t1);
//
//        return tensor;
//
//    }

//    static void resize_inplace(K_tensor t1, int[] sizes) {
//
//        t1.matrix = K_math.resize(t1.matrix, sizes);
//
//    }

//    static K_tensor transpose(K_tensor t1) { // todo : not tested.
//
//        K_tensor tensor = new K_tensor(K_math.transpose(t1.matrix));
//
//        define_as_same(tensor, t1);
//
//        return tensor;
//
//    }

//    static void transpose_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.transpose(t1.matrix);
//
//    }

    static K_tensor sum(K_tensor t1, int dim) {

        K_tensor tensor = new K_tensor(K_math.sum(t1.matrix, dim));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0], size_p1[1]));

        return tensor;

    }


    // special operations

    static K_tensor pow(K_tensor t1, double pow) {

        K_tensor tensor = new K_tensor(K_math.pow(t1.matrix, pow));

        define_as_child(tensor, t1);

        tensor.parent_grads.add(K_math.mul_scalar(pow, K_math.pow(t1.matrix, pow-1)));

        return tensor;

    }

//    static void pow_inplace(K_tensor t1, double pow) {
//
//        t1.matrix = K_math.pow(t1.matrix, pow);
//
//    }

    static K_tensor exp(K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.exp(t1.matrix));

        define_as_child(tensor, t1);

        tensor.parent_grads.add(tensor.matrix);

        return tensor;

    }

    static K_tensor log(K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.log(t1.matrix));

        define_as_child(tensor, t1);

        tensor.parent_grads.add(K_math.div_scalar(1, tensor.matrix));

        return tensor;

    }

//    static void exp_inplace(K_tensor t1, double pow) {
//
//        t1.matrix = K_math.exp(t1.matrix);
//
//    }

    static K_tensor tanh(K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.tanh(t1.matrix));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        Double[][] parent_grad = new Double[size_p1[0]][size_p1[1]];

        Double[] row1, row2;

        for (int i = 0; i < size_p1[0]; i++) {
            row1 = parent_grad[i];
            row2 = tensor.matrix[i];
            for (int j = 0; j < size_p1[1]; j++)
                row1[j] = 1 - Math.pow(row2[j], 2);

        }

        tensor.parent_grads.add(parent_grad);

        return tensor;

    }

//    static void tanh_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.tanh(t1.matrix);
//
//    }

    static K_tensor sigm(K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.sigm(t1.matrix));

        define_as_child(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        Double[][] parent_grad = new Double[size_p1[0]][size_p1[1]];

        Double[] row1, row2;

        for (int i = 0; i < size_p1[0]; i++) {
            row1 = parent_grad[i];
            row2 = tensor.matrix[i];
            for (int j = 0; j < size_p1[1]; j++)
                row1[j] = row2[j] * (1 - row2[j]);

        }

        tensor.parent_grads.add(parent_grad);

        return tensor;

    }

//    static void sigm_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.sigm(t1.matrix);
//
//    }

    static K_tensor mean_square(K_tensor t_lbl, K_tensor t_out) {

        return pow(sub(t_lbl, t_out), 2);

    }

    static K_tensor mean_square(Double[][] t_lbl, K_tensor t_out) {

        return pow(sub(t_lbl, t_out), 2);

    } static K_tensor mean_square(K_tensor t_out, Double[][] t_lbl) { return mean_square(t_lbl, t_out); }

//    static K_tensor softmax(K_tensor t1) { // stable softmax ; x - np.max(x) first.
//
//        K_tensor exp = exp(t1);
//
//        K_tensor exp_sum = scalar_tensor(sum(sum(exp, 0), 1), size(t1));
//
//        return div(exp, exp_sum);
//
//    }

    static K_tensor cross_entropy(K_tensor t_lbl, K_tensor t_out) {

        return mul(-1.0, mul(t_lbl, log(t_out)));

    }

    static K_tensor cross_entropy(Double[][] t_lbl, K_tensor t_out) {

        return mul(-1.0, mul(t_lbl, log(t_out)));

    }

//    static K_tensor softmax_cross_entropy(K_tensor t_lbl, K_tensor t_out) {
//
//        return cross_entropy(t_lbl, softmax(t_out));
//
//    }

//    static void softmax_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.softmax(t1.matrix);
//
//    }

}