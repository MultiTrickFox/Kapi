import java.util.ArrayList;
import java.util.Random;

// 2DO: compile as .jar

class K_base {

    static Random random = new Random(System.currentTimeMillis());

    static double mean = 0;
    static double dev = 1;
    
    // constructor
    
    K_base() { } static K_base instance = new K_base();
    
    // matrix initializers
    
    static Double[][] zeros(int hm_rows, int hm_cols) {
      
        Double[][] out = new Double[hm_rows][hm_cols];
        
        for (int i = 0; i < hm_rows; i++) {
            Double[] row = out[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = 0.0;
        }

        return out;
        
    }
    
    static Double[][] ones(int hm_rows, int hm_cols) {
      
        Double[][] out = new Double[hm_rows][hm_cols];
        
        for (int i = 0; i < hm_rows; i++) {
            Double[] row = out[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = 1.0;
        }
        
        return out;
        
    }
    
    static Double[][] randn(int hm_rows, int hm_cols) {
      
        Double[][] out = new Double[hm_rows][hm_cols];
        
        for (int i = 0; i < hm_rows; i++) {
            Double[] row = out[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = random.nextGaussian() * dev + mean;
        }
        
        return out;
        
    }
    
    // matrix operations

    static Double[][] add(Double[][] a, Double[][] b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a, row_b;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            row_b = b[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] + row_b[j];
        }
        
        return out;
        
    }

    static Double[][] sub(Double[][] a, Double[][] b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a, row_b;
              
        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            row_b = b[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] - row_b[j];
        }

        return out;
        
    }
    
    static Double[][] mul(Double[][] a, Double[][] b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a, row_b;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            row_b = b[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] * row_b[j];
        }

        return out;
        
    }
    
    static Double[][] div(Double[][] a, Double[][] b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a, row_b;
      
        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            row_b = b[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] / row_b[j];
        }
        
        return out;
        
    }
    
    static Double[][] matmul(Double[][] a, Double[][] b) {
      
        int hm_rows1 = a.length;
        int hm_cols1 = a[0].length;
        int hm_rows2 = b.length;
        int hm_cols2 = b[0].length;
        Double[][] out = new Double[hm_rows1][hm_cols2];

        assert hm_cols1 == hm_rows2;

        Double[] row, row_a;
        
        for (int i = 0; i < hm_rows1; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols2; j++) {
                row[j] = 0.0;
                for (int k = 0; k < hm_cols1; k++)
                    row[j] += row_a[k] * b[k][j];
            }
        }
                
        return out;
        
    }
    
    // scalar operations
    
    static Double[][] mul_scalar(double b, Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a;
      
        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] * b;
        }
                
        return out;
        
    } static Double[][] mul_scalar(Double[][] a, double b) { return mul_scalar(b, a); }
    
    static Double[][] sub_scalar(double b, Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a;
                
        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = b - row_a[j];
        }

        return out;
        
    }
    
    static Double[][] sub_scalar(Double[][] a, double b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] - b;
        }
        
        return out;
        
    }
    
    static Double[][] add_scalar(double b, Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] + b;
        }

        return out;
        
    } static Double[][] add_scalar(Double[][] a, double b) { return add_scalar(b, a); }
    
    // helpers

    static int size(Double[][] matrix, int dim) {

        if (dim == 0) return matrix.length;
        else          return matrix[0].length;

    } static int[] size(Double[][] matrix) { return new int[]{matrix.length,matrix[0].length}; }
    
    static Double[][] resize(Double[][] matrix, int[] sizes) {
     
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[sizes[0]][sizes[1]];
        
        int ctr = -1;
        for (int i = 0; i < sizes[0]; i++)
            for (int j = 0; j < sizes[1]; j++) {
              
                ctr++;
                // out[i][j] = a[][]; // TODO : derive row/col here.
              
            }
       
         return out;
       
    }
    
    static Double[] matrix2vector(Double[][] matrix) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[] out = new Double[matrix.length * matrix[0].length];

        int ctr = -1;
        for (Double[] row : matrix) {
            for (int j = 0; j < hm_cols; j++) {

                ctr++;
                out[ctr] = row[j];

            }
        }
        
         return out;
      
    }
    
    static Double[][] vector2matrix(Double[] vector, int[] sizes) {

        Double[][] out = new Double[sizes[0]][sizes[1]];

        int ctr = -1;
        for (Double[] row : out) {
            for (int j = 0; j < sizes[1]; j++) {

                ctr++;
                row[j] = vector[ctr];

            }
        }
        
         return out;
      
    }
    
    static Double[][] transpose(Double[][] matrix) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_cols][hm_rows];

        Double[] row;

        for (int i = 0; i < hm_cols; i++) {
            row = out[i];
            for (int j = 0; j < hm_rows; j++)
                row[j] = matrix[j][i];
        }

        return out;
        
    }

    static double sum(Double[][] matrix) {

        double sum = 0;

        for (Double[] row : matrix)
            for (Double col : row)
                sum += col;

        return sum;

    }
    
    // special operations

    static Double[][] exp(Double[][] matrix, double power) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = Math.pow(row_a[j], power);
        }
                
        return out;
        
    }

    static Double[][] sigm(Double[][] matrix) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = (1.0 / (1 + Math.exp(-row_a[j])));
        }

        return out;
       
    }
    
    static Double[][] tanh(Double[][] matrix) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = Math.tanh(row_a[j]);
        }

        return out;
       
    }

    static Double[][] cross_entropy(Double[][] target, Double[][] output) {

        int hm_rows = target.length;
        int hm_cols = target[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a, row_b;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = target[i];
            row_b = output[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = -(row_a[j]*Math.log(row_b[j])); // row[j] = (row_a[j] * Math.log(row_b[j])) + ((1 - row_a[j]) * Math.log(1 - row_b[j]));
        }

        return out;
   
    }
    
    static Double[][] softmax(Double[][] matrix) {
        
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        Double[] row, row_a;

        double sum = 0;
        for (int i = 0; i < hm_rows; i++) {
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                sum += Math.exp(row_a[j]);
        }
      
        if (hm_rows == 1) {

            row = out[0];
            row_a = matrix[0];

            for (int k = 0; k < hm_cols; k++)
                row[k] = Math.exp(row_a[k]) / sum;
        }
        
        if (hm_cols == 1)
            for (int k = 0; k < hm_rows; k++)
                out[k][0] = Math.exp(matrix[k][0])/sum;
        
        return out;
        
    }
    
    static Double[][] softmax(Double[][] matrix, int index_begin, int index_end) {
        
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        double sum = 0;
      
        if (hm_rows == 1) {

            Double[] row = out[0];
            Double[] row_a = matrix[0];

            for (int k = index_begin; k < index_end; k++)
                sum += Math.exp(row_a[k]);
          
            for (int k = index_begin; k < index_end; k++)
                row[k] = Math.exp(row_a[k])/sum;
           
        }
            
        if (hm_cols == 1) {
          
            for (int k = index_begin; k < index_end; k++)
                sum += Math.exp(matrix[k][0]);
          
            for (int k = index_begin; k < index_end; k++)
                out[k][0] = Math.exp(matrix[k][0])/sum;
          
        }
            
        return out;
        
    }

}


class Tensor{

    Double[][] matrix;
    Double[][] grads;

    ArrayList<Tensor> parents;
    ArrayList<Tensor> childs;

    // constructor

    Tensor(Double[][] matrix) {

        this.matrix = matrix;

        int[] size = K_base.instance.size(matrix);
        this.grads = new Double[size[0]][size[1]];
        for (int i = 0; i < size[0]; i++) {
            Double[] row = grads[i];
            for (int j = 0; j < size[1]; j++) {
                row[j] = 0.0;
            }
        }

        this.parents = new ArrayList<>();
        this.childs = new ArrayList<>();

    }

    // matrix initializers

    static Tensor zeros(int hm_rows, int hm_cols) {

        return new Tensor(K_base.zeros(hm_rows, hm_cols));

    }

    static Tensor ones(int hm_rows, int hm_cols) {

        return new Tensor(K_base.ones(hm_rows, hm_cols));

    }

    static Tensor randn(int hm_rows, int hm_cols) {

        return new Tensor(K_base.randn(hm_rows, hm_cols));

    }

    // matrix operations // TODO : _inplace this section

    static Tensor add(Tensor t1, Tensor t2) {

        Tensor tensor = new Tensor(K_base.add(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        return tensor;

    }

    static Tensor add(Tensor t1, Double[][] t2) {

        Tensor tensor = new Tensor(K_base.add(t1.matrix, t2));

        define_child_tensor(tensor, t1);

        return tensor;

    } static Tensor add(Double[][] t1, Tensor t2) { return add(t2, t1); }

    static Tensor sub(Tensor t1, Tensor t2) {

        Tensor tensor = new Tensor(K_base.sub(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        return tensor;

    }

    static Tensor sub(Tensor t1, Double[][] t2) {

        Tensor tensor = new Tensor(K_base.sub(t1.matrix, t2));

        define_child_tensor(tensor, t1);

        return tensor;

    } static Tensor sub(Double[][] t1, Tensor t2) { return sub(t2, t1); }

    static Tensor mul(Tensor t1, Tensor t2) {

        Tensor tensor = new Tensor(K_base.mul(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        return tensor;

    }

    static Tensor mul(Tensor t1, Double[][] t2) {

        Tensor tensor = new Tensor(K_base.mul(t1.matrix, t2));

        define_child_tensor(tensor, t1);

        return tensor;

    } static Tensor mul(Double[][] t1, Tensor t2) { return mul(t2, t1); }

    static Tensor div(Tensor t1, Tensor t2) {

        Tensor tensor = new Tensor(K_base.div(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        return tensor;

    }

    static Tensor div(Tensor t1, Double[][] t2) {

        Tensor tensor = new Tensor(K_base.div(t1.matrix, t2));

        tensor.parents.add(t1);

        t1.childs.add(tensor);

        return tensor;

    } static Tensor div(Double[][] t1, Tensor t2) { return div(t2, t1); }

    static Tensor matmul(Tensor t1, Tensor t2) {

        Tensor tensor = new Tensor(K_base.matmul(t1.matrix, t2.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        return tensor;

    }

    static Tensor matmul(Tensor t1, Double[][] t2) {

        Tensor tensor = new Tensor(K_base.matmul(t1.matrix, t2));

        define_child_tensor(tensor, t1);

        return tensor;

    } static Tensor matmul(Double[][] t1, Tensor t2) { return matmul(t2, t1); }

    // scalar operations // TODO : _inplace this section

    static Tensor add_scalar(Tensor t1, Double s) {

        Tensor tensor = new Tensor(K_base.add_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        return tensor;

    }

    static Tensor sub_scalar(Tensor t1, Double s) {

        Tensor tensor = new Tensor(K_base.sub_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        return tensor;

    }

    static Tensor sub_scalar(Double s, Tensor t1) {

        Tensor tensor = new Tensor(K_base.sub_scalar(s, t1.matrix));

        define_child_tensor(tensor, t1);

        return tensor;

    }

    static Tensor mul_scalar(Tensor t1, Double s) {

        Tensor tensor = new Tensor(K_base.mul_scalar(t1.matrix, s));

        define_child_tensor(tensor, t1);

        return tensor;

    }

    // helpers

    private static void define_same_tensor(Tensor t1, Tensor t2) {

        t1.childs.addAll(t2.childs);
        t1.parents.addAll(t2.parents);

        for (Tensor child : t2.childs)
            child.parents.add(t1);

        for (Tensor parent : t2.parents)
            parent.childs.add(t1);

    }

    private static void define_child_tensor(Tensor t_child, Tensor t_parent) {

        t_child.parents.add(t_parent);
        t_parent.childs.add(t_child);

    }

    static int[] size(Tensor t1) {

        return K_base.size(t1.matrix);

    }

    static int size(Tensor t1, int dim) {

        return K_base.size(t1.matrix, dim);

    }

    static Tensor resize(Tensor t1, int[] sizes) {

        Tensor tensor = new Tensor(K_base.resize(t1.matrix, sizes));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    static void resize_inplace(Tensor t1, int[] sizes) {

        t1.matrix = K_base.resize(t1.matrix, sizes);

    }

    static Tensor array2matrix(Double[] array, int[] sizes) {

        return new Tensor(K_base.vector2matrix(array, sizes));

    }

    static Tensor transpose(Tensor t1) {

        Tensor tensor = new Tensor(K_base.transpose(t1.matrix));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    static void transpose_inplace(Tensor t1) {

        t1.matrix = K_base.transpose(t1.matrix);

    }

    // special operations

    static Tensor exp(Tensor t1, double exp) {

        Tensor tensor = new Tensor(K_base.exp(t1.matrix, exp));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    static void exp_inplace(Tensor t1, double exp) {

        t1.matrix = K_base.exp(t1.matrix, exp);

    }

    static Tensor tanh(Tensor t1) {

        Tensor tensor = new Tensor(K_base.tanh(t1.matrix));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    static void tanh_inplace(Tensor t1) {

        t1.matrix = K_base.tanh(t1.matrix);

    }

    static Tensor sigm(Tensor t1) {

        Tensor tensor = new Tensor(K_base.tanh(t1.matrix));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    static void sigm_inplace(Tensor t1) {

        t1.matrix = K_base.sigm(t1.matrix);

    }

    static Tensor cross_entropy(Tensor t1, Tensor t2) {

        Tensor tensor = new Tensor(K_base.tanh(t1.matrix));

        define_child_tensor(tensor, t1);
        define_child_tensor(tensor, t2);

        return tensor;

    }

    static Tensor softmax(Tensor t1) {

        Tensor tensor = new Tensor(K_base.softmax(t1.matrix));

        define_same_tensor(tensor, t1);

        return tensor;

    }

    static void softmax_inplace(Tensor t1) {

        t1.matrix = K_base.softmax(t1.matrix);

    }

}
