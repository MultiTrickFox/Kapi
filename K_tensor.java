import java.util.ArrayList;


class K_tensor{

    Double[][] matrix;
    Double[][] grads;

    ArrayList<K_tensor> parents;
    ArrayList<K_tensor> childs;

    ArrayList<Double[][]> parent_grads;

    static ArrayList<K_tensor> graph = new ArrayList<>();

    // constructor

    K_tensor(Double[][] matrix) {

        this.matrix = matrix;

        int[] size = K_math.size(matrix);
        this.grads = new Double[size[0]][size[1]];
        for (int i = 0; i < size[0]; i++) {
            Double[] row = grads[i];
            for (int j = 0; j < size[1]; j++) {
                row[j] = 0.0;
            }
        }

        this.parents = new ArrayList<>();
        this.childs = new ArrayList<>();
        this.parent_grads = new ArrayList<>();

        graph.add(this);

    }

    K_tensor() { graph.add(this); }

    // matrix initializers

    static K_tensor zeros(int hm_rows, int hm_cols) {

        return new K_tensor(K_math.zeros(hm_rows, hm_cols));

    }

    static K_tensor ones(int hm_rows, int hm_cols) {

        return new K_tensor(K_math.ones(hm_rows, hm_cols));

    }

    static K_tensor randn(int hm_rows, int hm_cols) {

        return new K_tensor(K_math.randn(hm_rows, hm_cols));

    }

    // matrix operations

    static K_tensor add(K_tensor t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.add(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        int[] size_p1 = K_math.size(t1.matrix);
        int[] size_p2 = K_math.size(t2.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));
        tensor.parent_grads.add(K_math.ones(size_p2[0],size_p2[1]));

        return tensor;

    }

    static K_tensor add(K_tensor t1, Double[][] t2) {

        K_tensor tensor = new K_tensor(K_math.add(t1.matrix, t2));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    } static K_tensor add(Double[][] t1, K_tensor t2) { return add(t2, t1); }

    static K_tensor sub(K_tensor t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.sub(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        int[] size_p1 = K_math.size(t1.matrix);
        int[] size_p2 = K_math.size(t2.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));
        tensor.parent_grads.add(K_math.constant(size_p2[0],size_p2[1], -1));

        return tensor;

    }

    static K_tensor sub(K_tensor t1, Double[][] t2) {

        K_tensor tensor = new K_tensor(K_math.sub(t1.matrix, t2));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    } static K_tensor sub(Double[][] t1, K_tensor t2) { return sub(t2, t1); }

    static K_tensor mul(K_tensor t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.mul(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        tensor.parent_grads.add(t2.matrix);
        tensor.parent_grads.add(t1.matrix);

        return tensor;

    }

    static K_tensor mul(K_tensor t1, Double[][] t2) {

        K_tensor tensor = new K_tensor(K_math.mul(t1.matrix, t2));

        define_child_tensor(tensor, t1);

        tensor.parent_grads.add(t2);

        return tensor;

    } static K_tensor mul(Double[][] t1, K_tensor t2) { return mul(t2, t1); }

    static K_tensor div(K_tensor t1, K_tensor t2) {

        K_tensor tensor = new K_tensor(K_math.div(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        tensor.parent_grads.add(K_math.div_scalar(1, t2.matrix));
        tensor.parent_grads.add(t1.matrix);

        return tensor;

    }
//
//    static K_tensor div(K_tensor t1, Double[][] t2) {
//
//        K_tensor tensor = new K_tensor(K_math.div(t1.matrix, t2));
//
//        define_child_tensor(tensor, t1);
//
//        tensor.parent_grads.add(K_math.div_scalar(1, t2));
//
//        return tensor;
//
//    } static K_tensor div(Double[][] t1, K_tensor t2) { return div(t2, t1); }

    static K_tensor matmul(K_tensor t1, K_tensor t2) {

        Double[][][] results = K_math.matmul_wgrads(t1.matrix, t2.matrix);

        K_tensor tensor = new K_tensor(results[0]);

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        tensor.parent_grads.add(results[1]);
        tensor.parent_grads.add(results[2]);

        return tensor;

    }

    static K_tensor matmul(K_tensor t1, Double[][] t2) {

        Double[][][] results = K_math.matmul_wgrads(t1.matrix, t2);

        K_tensor tensor = new K_tensor(results[0]);

        define_child_tensor(tensor, t1);

        tensor.parent_grads.add(results[1]);

        return tensor;

    } static K_tensor matmul(Double[][] t1, K_tensor t2) { return matmul(t2, t1); }

    // scalar operations

    static K_tensor add(K_tensor t1, Double s) {

        K_tensor tensor = new K_tensor(K_math.add_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    } static K_tensor add(Double s, K_tensor t1) { return add(t1, s); }

    static K_tensor sub(K_tensor t1, Double s) {

        K_tensor tensor = new K_tensor(K_math.sub_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    }

    static K_tensor sub(Double s, K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.sub_scalar(s, t1.matrix));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.constant(size_p1[0],size_p1[1], -1));

        return tensor;

    }

    static K_tensor mul(K_tensor t1, Double s) {

        K_tensor tensor = new K_tensor(K_math.mul_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.constant(size_p1[0],size_p1[1], s));

        return tensor;

    } static K_tensor mul(Double s, K_tensor t1) { return mul(t1, s); }

//    static K_tensor div_scalar(Double s, K_tensor t1) {
//
//        K_tensor tensor = new K_tensor(K_math.div_scalar(s, t1.matrix));
//
//        define_child_tensor(tensor, t1);
//
//        int[] size_p1 = K_math.size(t1.matrix);
//
//        tensor.parent_grads.add(K_math.constant(size_p1[0], size_p1[1], ??))
//
//        return tensor;
//
//    }

    // graph helpers

    private static void define_same_tensor(K_tensor t1, K_tensor t2) { // if problematic, interchange w/ new-node w/ back_grad = ones()

        t1.childs.addAll(t2.childs);
        t1.parents.addAll(t2.parents);
        t1.parent_grads = t2.parent_grads;

        for (K_tensor child : t2.childs)
            child.parents.add(t1);

        for (K_tensor parent : t2.parents)
            parent.childs.add(t1);

        t1.grads = t2.grads;

    }

    private static void define_child_tensor(K_tensor t_child, K_tensor t_parent) {

        t_child.parents.add(t_parent);
        t_parent.childs.add(t_child);

    }

    static void fill(K_tensor t1) {

        Double[][] sum = K_math.constant(size(t1)[0], size(t1)[1], sum(sum(t1,0),1).matrix[0][0]);

        t1.grads = K_math.add(t1.grads, sum);

        int parent_ctr = -1;
        for (K_tensor parent : t1.parents) {
            parent_ctr++;

            fill(parent, t1.parent_grads.get(parent_ctr));

        }

        // TODO : fill in grads of parents wrt parent_grads

    }

    static void fill(K_tensor t1, Double[][] incoming) {

        t1.grads = K_math.add(t1.grads, incoming);

        int parent_ctr = -1;
        for (K_tensor parent : t1.parents) {
            parent_ctr++;

            fill(parent, t1.parent_grads.get(parent_ctr));

        }

    }

    static void empty() {

        for (K_tensor tensor : graph)

            tensor.grads = K_math.zeros(size(tensor, 0), size(tensor, 1));

        graph = new ArrayList<>();

    }

    static K_tensor scalar_tensor(K_tensor t1) {

        int[] sizes = K_math.size(t1.matrix);

        K_tensor tensor = new K_tensor(K_math.constant(sizes[0], sizes[1], t1.matrix[0][0]));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    // tensor helpers

    static int[] size(K_tensor t1) {

        return K_math.size(t1.matrix);

    }

    static int size(K_tensor t1, int dim) {

        return K_math.size(t1.matrix, dim);

    }

    static K_tensor resize(K_tensor t1, int[] sizes) { // todo : not tested.

        K_tensor tensor = new K_tensor(K_math.resize(t1.matrix, sizes));

        define_same_tensor(tensor, t1);

        return tensor;

    }

//    static void resize_inplace(K_tensor t1, int[] sizes) {
//
//        t1.matrix = K_math.resize(t1.matrix, sizes);
//
//    }

    static K_tensor array2tensor(Double[] array, int[] sizes) {

        return new K_tensor(K_math.vector2matrix(array, sizes));

    }

    static double[] tensor2array(K_tensor tensor) {

        int[] sizes = size(tensor);
        double[] array = new double[sizes[0]*sizes[1]];

        int ctr = -1;
        for (int i = 0; i < sizes[0]; i++)
            for (int j = 0; j < sizes[1]; j++) {
                ctr++;

                array[ctr] = tensor.matrix[i][j];

            }

        return array;

    }

    static K_tensor transpose(K_tensor t1) { // todo : not tested.

        K_tensor tensor = new K_tensor(K_math.transpose(t1.matrix));

        define_same_tensor(tensor, t1);

        return tensor;

    }

//    static void transpose_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.transpose(t1.matrix);
//
//    }

    static K_tensor sum(K_tensor t1, int dim) {

        K_tensor tensor = new K_tensor(K_math.sum(t1.matrix, dim));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0], size_p1[1]));

        return tensor;
    }

    // special operations

    static K_tensor pow(K_tensor t1, double pow) {

        K_tensor tensor = new K_tensor(K_math.pow(t1.matrix, pow));

        define_child_tensor(tensor, t1);

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

        define_child_tensor(tensor, t1);

        tensor.parent_grads.add(tensor.matrix);

        return tensor;

    }

    static K_tensor log(K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.log(t1.matrix));

        define_child_tensor(tensor, t1);

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

        define_child_tensor(tensor, t1);

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

        K_tensor tensor = new K_tensor(K_math.tanh(t1.matrix));

        define_child_tensor(tensor, t1);

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

    static K_tensor cross_entropy(K_tensor t1, K_tensor t2) {

        return mul(-1.0, mul(t1, log(t2)));

    }

    static K_tensor softmax(K_tensor t1, int dim, int true_index) { // stable softmax ; x - np.max(x) first.

        assert t1.matrix.length == 1 || t1.matrix[0].length == 1;

        K_tensor exp = exp(t1);

        K_tensor exp_sum = scalar_tensor(sum(sum(exp, 0), 1));

        return div(exp, exp_sum);

    }

//    static void softmax_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.softmax(t1.matrix);
//
//    }

}