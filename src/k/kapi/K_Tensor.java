package k.kapi;

import java.util.ArrayList;


class K_Tensor {


    Float[][] matrix;
    Float[][] grad;

    ArrayList<K_Tensor> parents;
    ArrayList<K_Tensor> childs;

    ArrayList<Float[][]> parent_grads;

    boolean requires_grad;

    String type;


    //static ArrayList<K_Tensor> graph = new ArrayList<>();


    // constructor

    K_Tensor(Float[][] matrix) {

        this.matrix = matrix;

        int[] size = K_Math.size(matrix);
        this.grad = K_Math.zeros(size[0], size[1]);

        this.parents = new ArrayList<>();
        this.childs = new ArrayList<>();
        this.parent_grads = new ArrayList<>();

        this.type = "tensor";
        this.requires_grad = false;

        //graph.add(this);

    }

    K_Tensor() { } static K_Tensor instance = new K_Tensor();


    // matrix initializers

    static K_Tensor zeros(int hm_rows, int hm_cols) {

        K_Tensor tensor = new K_Tensor(K_Math.zeros(hm_rows, hm_cols));

        tensor.requires_grad = true;

        return tensor;

    } static K_Tensor zeros(int[] sizes) { return zeros(sizes[0], sizes[1]); } // todo : do for others.

    static K_Tensor ones(int hm_rows, int hm_cols) {

        K_Tensor tensor = new K_Tensor(K_Math.ones(hm_rows, hm_cols));

        tensor.requires_grad = true;

        return tensor;

    }

    static K_Tensor randn(int hm_rows, int hm_cols) {

        K_Tensor tensor = new K_Tensor(K_Math.randn(hm_rows, hm_cols));

        tensor.requires_grad = true;

        return tensor;

    }

    static K_Tensor identity(int hm_rows, int hm_cols) {

        K_Tensor tensor = new K_Tensor(K_Math.identity(hm_rows, hm_cols));

        tensor.requires_grad = true;

        return tensor;

    }

    static K_Tensor constants(int hm_rows, int hm_cols, float val) {

        K_Tensor tensor = new K_Tensor(K_Math.constants(hm_rows, hm_cols, val));

        tensor.requires_grad = true;

        return tensor;

    }


    // matrix operations

    static K_Tensor add(K_Tensor t1, K_Tensor t2) {

        K_Tensor tensor = new K_Tensor(K_Math.add(t1.matrix, t2.matrix));

        define_as_child(tensor, t1);
        define_as_child(tensor, t2);

        int[] size_p1 = K_Math.size(t1.matrix);
        int[] size_p2 = K_Math.size(t2.matrix);

        tensor.parent_grads.add(K_Math.ones(size_p1[0],size_p1[1]));
        tensor.parent_grads.add(K_Math.ones(size_p2[0],size_p2[1]));

        return tensor;

    }

    static K_Tensor add(K_Tensor t1, Float[][] t2) {

        K_Tensor tensor = new K_Tensor(K_Math.add(t1.matrix, t2));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        tensor.parent_grads.add(K_Math.ones(size_p1[0],size_p1[1]));

        return tensor;

    } static K_Tensor add(Float[][] t1, K_Tensor t2) { return add(t2, t1); }

    static K_Tensor sub(K_Tensor t1, K_Tensor t2) {

        K_Tensor tensor = new K_Tensor(K_Math.sub(t1.matrix, t2.matrix));

        define_as_child(tensor, t1);
        define_as_child(tensor, t2);

        int[] size_p1 = K_Math.size(t1.matrix);
        int[] size_p2 = K_Math.size(t2.matrix);

        tensor.parent_grads.add(K_Math.ones(size_p1[0],size_p1[1]));
        tensor.parent_grads.add(K_Math.constants(size_p2[0],size_p2[1], -1));

        return tensor;

    }

    static K_Tensor sub(K_Tensor t1, Float[][] t2) {

        K_Tensor tensor = new K_Tensor(K_Math.sub(t1.matrix, t2));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        tensor.parent_grads.add(K_Math.ones(size_p1[0],size_p1[1]));

        return tensor;

    }

    static K_Tensor sub(Float[][] t2, K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(K_Math.sub(t1.matrix, t2));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        tensor.parent_grads.add(K_Math.ones(size_p1[0],size_p1[1]));

        return tensor;

    }

    static K_Tensor mul(K_Tensor t1, K_Tensor t2) {

        K_Tensor tensor = new K_Tensor(K_Math.mul(t1.matrix, t2.matrix));

        define_as_child(tensor, t1);
        define_as_child(tensor, t2);

        tensor.parent_grads.add(t2.matrix);
        tensor.parent_grads.add(t1.matrix);

        return tensor;

    }

    static K_Tensor mul(K_Tensor t1, Float[][] t2) {

        K_Tensor tensor = new K_Tensor(K_Math.mul(t1.matrix, t2));

        define_as_child(tensor, t1);

        tensor.parent_grads.add(t2);

        return tensor;

    } static K_Tensor mul(Float[][] t1, K_Tensor t2) { return mul(t2, t1); }

    static K_Tensor div(K_Tensor t1, K_Tensor t2) {

        return mul(t1, pow(t2, -1));

    }

    static K_Tensor div(K_Tensor t1, Float[][] t2) {

        return mul(t1, K_Math.pow(t2, -1));

    }

    static K_Tensor div(Float[][] t2, K_Tensor t1) {

        return mul(t2, pow(t1, -1));

    }

    static K_Tensor matmul(K_Tensor t1, K_Tensor t2) {

        K_Tensor tensor = new K_Tensor(K_Math.matmul(t1.matrix, t2.matrix));
        tensor.type = "matmul";

        define_as_child(tensor, t1);
        define_as_child(tensor, t2);

        tensor.parent_grads.add(t2.matrix);
        tensor.parent_grads.add(t1.matrix);

        return tensor;

    }

    static K_Tensor matmul(K_Tensor t1, Float[][] t2) {

        K_Tensor tensor = new K_Tensor(K_Math.matmul(t1.matrix, t2));
        tensor.type = "matmul";

        define_as_child(tensor, t1);

        tensor.parent_grads.add(t2);

        return tensor;

    }

    static K_Tensor matmul(Float[][] t1, K_Tensor t2) {

        K_Tensor tensor = new K_Tensor(K_Math.matmul(t1, t2.matrix));
        tensor.type = "matmul";

        define_as_child(tensor, t2);

        tensor.parent_grads.add(t1);

        return tensor;

    }


    // scalar operations

    static K_Tensor add(K_Tensor t1, Float s) {

        K_Tensor tensor = new K_Tensor(K_Math.add_scalar(t1.matrix, s));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        tensor.parent_grads.add(K_Math.ones(size_p1[0],size_p1[1]));

        return tensor;

    } static K_Tensor add(Float s, K_Tensor t1) { return add(t1, s); }

    static K_Tensor sub(K_Tensor t1, Float s) {

        return add(t1, -s);

    }

    static K_Tensor sub(Float s, K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(K_Math.sub_scalar(s, t1.matrix));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        tensor.parent_grads.add(K_Math.constants(size_p1[0],size_p1[1], -1));

        return tensor;

    }

    static K_Tensor mul(K_Tensor t1, Float s) {

        K_Tensor tensor = new K_Tensor(K_Math.mul_scalar(t1.matrix, s));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        tensor.parent_grads.add(K_Math.constants(size_p1[0],size_p1[1], s));

        return tensor;

    } static K_Tensor mul(Float s, K_Tensor t1) { return mul(t1, s); }

    static K_Tensor div(Float s, K_Tensor t1) {

        return mul(s, pow(t1, -1));

    } static K_Tensor div(K_Tensor t1, Float s) { return mul(t1, 1/s); }


    // graph helpers

    private static void define_as_same(K_Tensor t1, K_Tensor t2) {

        t1.childs.addAll(t2.childs);
        t1.parents.addAll(t2.parents);

        for (K_Tensor child : t2.childs)
            child.parents.add(t1);

        for (K_Tensor parent : t2.parents)
            parent.childs.add(t1);

    }

    private static void define_as_child(K_Tensor t1, K_Tensor t2) {

        t1.parents.add(t2);
        t2.childs.add(t1);

    }

    static float fill_grads(K_Tensor t1) {

        Float[][] incoming = K_Math.ones(K_Math.size(t1.matrix, 0), K_Math.size(t1.matrix, 1));

        fill_grads(t1, incoming);

        return K_Math.sum(t1.matrix);

    }

    private static void fill_grads(K_Tensor t1, Float[][] incoming) {

        if (t1.requires_grad)
            t1.grad = K_Math.add(t1.grad, incoming);

        switch (t1.type) {

            case "matmul": {

                int parent_ctr = -1;
                for (K_Tensor parent : t1.parents) {
                    parent_ctr++;

                    fill_grads(parent, matmul_backwards(t1.parent_grads.get(parent_ctr), incoming, parent_ctr == 0 ? "A" : "B"));

                }

                break;
            }

            case "resize": {

                K_Tensor parent = t1.parents.get(0);

                fill_grads(parent, K_Math.resize(incoming, size(parent)));

                break;
            }

            case "transpose": {

                K_Tensor parent = t1.parents.get(0);

                fill_grads(parent, K_Math.transpose(incoming));

                break;
            }

            default: {

                int parent_ctr = -1;
                for (K_Tensor parent : t1.parents) {
                    parent_ctr++;

                    fill_grads(parent, K_Math.mul(t1.parent_grads.get(parent_ctr), incoming));

                }

            }

        }

    }

//    static void empty_grads() {
//
//        for (K_Tensor tensor : graph)
//
//            tensor.grad = K_Math.zeros(size(tensor, 0), size(tensor, 1));
//
//    }

//    static void release_graph() {
//
//        for (K_Tensor tensor : graph) {
//
//            tensor.parents = new ArrayList<>();
//            tensor.childs = new ArrayList<>();
//            tensor.parent_grads = new ArrayList<>();
//
//        }
//
//        graph = new ArrayList<>();
//
//    }

    private static Float[][] matmul_backwards(Float[][] parent_grad, Float[][] incoming_grad, String for_matrix) {

        int grad_rows = parent_grad.length;
        int grad_cols = parent_grad[0].length;
        int incoming_rows = incoming_grad.length;
        int incoming_cols = incoming_grad[0].length;

        if (for_matrix.equals("A")) {

            int outgoing_rows = incoming_rows;
            int outgoing_cols = grad_rows;
            Float[][] out = new Float[outgoing_rows][outgoing_cols];

            Float[] incoming_row, parent_row, outgoing_row;

            for (int i = 0; i < outgoing_rows; i++) {

                outgoing_row = out[i];
                incoming_row = incoming_grad[i];

                for (int j = 0; j < outgoing_cols; j++) {

                    parent_row = parent_grad[j];

                    outgoing_row[j] = K_Math.vector_sum(K_Math.vector_mul(incoming_row, parent_row));

                }

            }

            return out;

        }

        if (for_matrix.equals("B")) {

            int outgoing_rows = grad_cols;
            int outgoing_cols = incoming_cols;
            Float[][] out = new Float[outgoing_rows][outgoing_cols];

            Float[] incoming_row, parent_row, outgoing_row;

            incoming_grad = K_Math.transpose(incoming_grad);
            parent_grad = K_Math.transpose(parent_grad);

            for (int i = 0; i < outgoing_rows; i++) {

                outgoing_row = out[i];
                parent_row = parent_grad[i];

                for (int j = 0; j < outgoing_cols; j++) {

                    incoming_row = incoming_grad[j];

                    outgoing_row[j] = K_Math.vector_sum(K_Math.vector_mul(incoming_row, parent_row));

                }

            }

            return out;

        }

        return null;

    }


    // tensor helpers

    static int[] size(K_Tensor t1) {

        return K_Math.size(t1.matrix);

    }

    static int size(K_Tensor t1, int dim) {

        return K_Math.size(t1.matrix, dim);

    }

    static K_Tensor resize(K_Tensor t1, int[] sizes) {

        K_Tensor tensor = new K_Tensor(K_Math.resize(t1.matrix, sizes));

        define_as_child(tensor, t1);

        tensor.type = "resize";

        tensor.parent_grads.add(null);

        return tensor;

    } static K_Tensor resize(K_Tensor t1, int size1, int size2) { return resize(t1, new int[]{size1, size2}); }

    static K_Tensor transpose(K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(K_Math.transpose(t1.matrix));

        define_as_child(tensor, t1);

        tensor.type = "transpose";

        tensor.parent_grads.add(null);

        return tensor;

    }

    static K_Tensor sum(K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(new Float[][]{{K_Math.sum(t1.matrix)}});

        define_as_child(tensor, t1);

        int[] size_t1 = K_Math.size(t1.matrix);

        tensor.parent_grads.add(K_Math.ones(size_t1[0], size_t1[1]));

        return tensor;

    }


    // special operations

    static K_Tensor pow(K_Tensor t1, float pow) {

        K_Tensor tensor = new K_Tensor(K_Math.pow(t1.matrix, pow));

        define_as_child(tensor, t1);

        tensor.parent_grads.add(K_Math.mul_scalar(pow, K_Math.pow(t1.matrix, pow-1)));

        return tensor;

    }

    static K_Tensor exp(K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(K_Math.exp(t1.matrix));

        define_as_child(tensor, t1);

        tensor.parent_grads.add(tensor.matrix);

        return tensor;

    }

    static K_Tensor log(K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(K_Math.log(t1.matrix));

        define_as_child(tensor, t1);

        tensor.parent_grads.add(K_Math.div_scalar(1, tensor.matrix));

        return tensor;

    }

    static K_Tensor tanh(K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(K_Math.tanh(t1.matrix));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        Float[][] parent_grad = new Float[size_p1[0]][size_p1[1]];

        Float[] row1, row2;

        for (int i = 0; i < size_p1[0]; i++) {
            row1 = parent_grad[i];
            row2 = tensor.matrix[i];
            for (int j = 0; j < size_p1[1]; j++)
                row1[j] = (float) (1 - Math.pow(row2[j], 2));

        }

        tensor.parent_grads.add(parent_grad);

        return tensor;

    }

    static K_Tensor sigm(K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(K_Math.sigm(t1.matrix));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        Float[][] parent_grad = new Float[size_p1[0]][size_p1[1]];

        Float[] row1, row2;

        for (int i = 0; i < size_p1[0]; i++) {
            row1 = parent_grad[i];
            row2 = tensor.matrix[i];
            for (int j = 0; j < size_p1[1]; j++)
                row1[j] = row2[j] * (1 - row2[j]);

        }

        tensor.parent_grads.add(parent_grad);

        return tensor;

    }

    static K_Tensor elu(K_Tensor t1) {

        K_Tensor tensor = new K_Tensor(K_Math.elu(t1.matrix));

        define_as_child(tensor, t1);

        int[] size_p1 = K_Math.size(t1.matrix);

        Float[][] parent_grad = new Float[size_p1[0]][size_p1[1]];

        Float[] row1, row2;

        for (int i = 0; i < size_p1[0]; i++) {
            row1 = parent_grad[i];
            row2 = tensor.matrix[i];
            for (int j = 0; j < size_p1[1]; j++)
                row1[j] = row2[j] >= 0 ? 1 : row2[j] +1;

                //row1[j] = row2[j] >= 0 ? row2[j] : (float) Math.exp(row2[j]) -1;

        }

        tensor.parent_grads.add(parent_grad);

        return tensor;

    }

    static K_Tensor mean_square(K_Tensor t_lbl, K_Tensor t_out) {

        return pow(sub(t_lbl, t_out), 2);

    }

    static K_Tensor mean_square(Float[][] t_lbl, K_Tensor t_out) {

        return pow(sub(t_lbl, t_out), 2);

    } static K_Tensor mean_square(K_Tensor t_out, Float[][] t_lbl) { return mean_square(t_lbl, t_out); }

//    static K_tensor softmax(K_tensor t1) { // stable softmax ; x - np.max(x) first. // TODO: open up softmax & cross entropy
//
//        K_tensor exp = exp(t1);
//
//        int[] size_t1 = K_Math.size(t1.matrix);
//        K_tensor sum_exp = constants(size_t1[0], size_t1[1], K_Math.sum(t1.matrix));
//
//        define_as_child(sum_exp, t1);
//
//        sum_exp.parent_grads.add(K_Math.ones(size_t1[0], size_t1[1]));
//
//        return div(exp, sum_exp);
//
//    }
//
//    static K_tensor cross_entropy(K_tensor t_lbl, K_tensor t_out) {
//
//        return mul(-1.0, mul(t_lbl, log(t_out)));
//
//    }
//
//    static K_tensor cross_entropy(Float[][] t_lbl, K_tensor t_out) {
//
//        return mul(-1.0, mul(t_lbl, log(t_out)));
//
//    }
//
//    static K_tensor softmax_cross_entropy(K_tensor t_lbl, K_tensor t_out) {
//
//        return cross_entropy(t_lbl, softmax(t_out));
//
//    }


}
