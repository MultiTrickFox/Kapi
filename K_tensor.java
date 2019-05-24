import java.util.ArrayList;


class K_tensor{

    Double[][] matrix;
    Double[][] grads;

    ArrayList<K_tensor> parents;
    ArrayList<K_tensor> childs;

    ArrayList<Double[][]> parent_grads;

    String type;

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
        this.type = "t";

    }

    K_tensor(String type) {

        this.parents = new ArrayList<>();
        this.childs = new ArrayList<>();
        this.parent_grads = new ArrayList<>();
        this.type = type;

    }

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

        tensor.parent_grads.add(K_math.div_scalar(1, t2.matrix));  // TODO : correct.
        tensor.parent_grads.add(t1.matrix);          // TODO : correct.

        return tensor;

    }

    static K_tensor div(K_tensor t1, Double[][] t2) {

        K_tensor tensor = new K_tensor(K_math.div(t1.matrix, t2));

        define_child_tensor(tensor, t1);

        tensor.parent_grads.add(K_math.div_scalar(1, t2));

        return tensor;

    } static K_tensor div(Double[][] t1, K_tensor t2) { return div(t2, t1); }

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

    static K_tensor add_scalar(K_tensor t1, Double s) {

        K_tensor tensor = new K_tensor(K_math.add_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    }

    static K_tensor sub_scalar(K_tensor t1, Double s) {

        K_tensor tensor = new K_tensor(K_math.sub_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.ones(size_p1[0],size_p1[1]));

        return tensor;

    }

    static K_tensor sub_scalar(Double s, K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.sub_scalar(s, t1.matrix));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.constant(size_p1[0],size_p1[1], -1));

        return tensor;

    }

    static K_tensor mul_scalar(K_tensor t1, Double s) {

        K_tensor tensor = new K_tensor(K_math.mul_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        int[] size_p1 = K_math.size(t1.matrix);

        tensor.parent_grads.add(K_math.constant(size_p1[0],size_p1[1], s));

        return tensor;

    }

    static K_tensor div_scalar(Double s, K_tensor t1) { //todo : left here.

        K_tensor tensor = new K_tensor(K_math.div_scalar(s, t1.matrix));

        define_child_tensor(tensor, t1);

        return tensor;

    }

    // helpers

    private static void define_same_tensor(K_tensor t1, K_tensor t2) {

        t1.childs.addAll(t2.childs);
        t1.parents.addAll(t2.parents);

        for (K_tensor child : t2.childs)
            child.parents.add(t1);

        for (K_tensor parent : t2.parents)
            parent.childs.add(t1);

    }

    private static void define_child_tensor(K_tensor t_child, K_tensor t_parent) {

        t_child.parents.add(t_parent);
        t_parent.childs.add(t_child);

    }

//    private static Double[][] grad_wrt(K_tensor t1, K_tensor t2) {
//
//        int[] grad_size = K_math.size(t1.matrix);
//        Double[][] grad = new Double[grad_size[0]][grad_size[1]];
//
//        switch(t2.type) {
//
//            case "t":
//
//            case "sigm":
//
//            case "tanh":
//
//            case "exp":
//
//            case "soft":
//
//        }
//
//    }

    static void backward(K_tensor t1) {

        // todo : fill in grads by taking grad_wrt

    }

    static int[] size(K_tensor t1) {

        return K_math.size(t1.matrix);

    }

    static int size(K_tensor t1, int dim) {

        return K_math.size(t1.matrix, dim);

    }

    static K_tensor resize(K_tensor t1, int[] sizes) {

        K_tensor tensor = new K_tensor(K_math.resize(t1.matrix, sizes));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    static void resize_inplace(K_tensor t1, int[] sizes) {

        t1.matrix = K_math.resize(t1.matrix, sizes);

    }

    static K_tensor array2matrix(Double[] array, int[] sizes) {

        return new K_tensor(K_math.vector2matrix(array, sizes));

    }

    static K_tensor transpose(K_tensor t1) {

        K_tensor tensor = new K_tensor(K_math.transpose(t1.matrix));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    static void transpose_inplace(K_tensor t1) {

        t1.matrix = K_math.transpose(t1.matrix);

    }

    // special operations

    static K_tensor exp(K_tensor t1, double exp) {

        K_tensor tensor = new K_tensor(K_math.exp(t1.matrix, exp));

        // define_same_tensor(tensor, t1);

//        int[] size = K_math.size(t1.matrix);
//        double back =
//
//                K_tensor t_exp = new K_tensor(K_math.identity(size[0], size[1], back));
//
//        define_child_tensor(t_exp);

        return tensor;

    }

//    static void exp_inplace(K_tensor t1, double exp) {
//
//        t1.matrix = K_math.exp(t1.matrix, exp);
//
//    }

    static K_tensor tanh(K_tensor t1) { // TODO : revisit.

        K_tensor tensor = new K_tensor(K_math.tanh(t1.matrix));

        // define_same_tensor(tensor, t1);

        return tensor;

    }

//    static void tanh_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.tanh(t1.matrix);
//
//    }

    static K_tensor sigm(K_tensor t1) { // TODO : revisit.

        K_tensor tensor = new K_tensor(K_math.tanh(t1.matrix));

        // define_same_tensor(tensor, t1);



        return tensor;

    }

//    static void sigm_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.sigm(t1.matrix);
//
//    }

    static K_tensor cross_entropy(K_tensor t1, K_tensor t2) { // TODO : revisit.

        K_tensor tensor = new K_tensor(K_math.cross_entropy(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        return tensor;

    }

    static K_tensor softmax(K_tensor t1) { // TODO : revisit.

        K_tensor tensor = new K_tensor(K_math.softmax(t1.matrix));

        define_same_tensor(tensor, t1);

        return tensor;

    }

//    static void softmax_inplace(K_tensor t1) {
//
//        t1.matrix = K_math.softmax(t1.matrix);
//
//    }

}


class K_graph{

    static void define_child_node(Node n1, Node n2) {

        n1.parents.add(n2);
        n2.childs.add(n1);

    }

    class Node{

        ArrayList<Node> parents;
        ArrayList<Node> childs;

        String type;
        K_tensor tensor;

        Node(String type) {

            this.type = type;
            this.parents = new ArrayList<>();

        }

        Node(String type, K_tensor tensor) {

            this.type = type;
            this.tensor = tensor;
            this.parents = new ArrayList<>();
            this.childs = new ArrayList<>();

        }

        Double[][] backward(Double[][] incoming) {

            switch(this.type) {

                case "t":

                case "sigm":

                case "tanh":

                case "exp":

                case "soft":



            }

        }

    }


}