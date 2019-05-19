import java.util.Arrays;
import java.util.Random;

// 2DO: compile as .jar

class K_base {

    Random random = new Random(System.currentTimeMillis());

    double mean = 0;
    double dev = 1;
    
    // constructor
    
    K_base() {

        this.random = new Random(System.currentTimeMillis());

    }
    
    // matrix initializers
    
    Double[][] zeros(int hm_rows, int hm_cols) {
      
        Double[][] out = new Double[hm_rows][hm_cols];
        
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = 0.0;
        
        return out;
        
    }
    
    Double[][] ones(int hm_rows, int hm_cols) {
      
        Double[][] out = new Double[hm_rows][hm_cols];
        
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = 1.0;
        
        return out;
        
    }
    
    Double[][] randn(int hm_rows, int hm_cols) {
      
        Double[][] out = new Double[hm_rows][hm_cols];
        
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = random.nextGaussian() * dev + mean;
        
        return out;
        
    }
    
    // matrix operations

    Double[][] add(Double[][] a, Double[][] b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = a[i][j] + b[i][j];
        
        return out;
        
    }

    Double[][] sub(Double[][] a, Double[][] b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
              
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = a[i][j] - b[i][j];

        return out;
        
    }
    
    Double[][] mul(Double[][] a, Double[][] b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = a[i][j] * b[i][j];

        return out;
        
    }
    
    Double[][] div(Double[][] a, Double[][] b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
      
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = a[i][j] / b[i][j];
        
        return out;
        
    }
    
    Double[][] matmul(Double[][] a, Double[][] b) {
      
        int hm_rows1 = a.length;
        int hm_cols1 = a[0].length;
        int hm_rows2 = b.length;
        int hm_cols2 = b[0].length;
        Double[][] out = new Double[hm_rows1][hm_cols2];
        
        for (int i = 0; i < hm_rows1; i++)
            for (int j = 0; j < hm_cols2; j++) {
              
                  out[i][j] = 0.0;
                  for (int k = 0; k < hm_cols1; k++)
                      out[i][j] += a[i][k] * b[k][j];
              
            }
                
        return out;
        
    }
    
    // scalar operations
    
    Double[][] mul_scalar(double b, Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
      
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = a[i][j] * b;
                
        return out;
        
    }; Double[][] mul_scalar(Double[][] a, double b) { return mul_scalar(b, a); }
    
    Double[][] sub_scalar(double b, Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
                
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = b - a[i][j];

        return out;
        
    }
    
    Double[][] sub_scalar(Double[][] a, double b) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = a[i][j] - b;
        
        return out;
        
    }
    
    Double[][] add_scalar(double b, Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = a[i][j] + b;

        return out;
        
    }; Double[][] add_scalar(Double[][] a, double b) { return add_scalar(b, a); }
    
    // helper operations

    int size(Double[][] matrix, int dim) {

        if (dim == 0) return matrix.length;
        else          return matrix[0].length;

    }; int[] size(Double[][] matrix) { return new int[]{matrix.length,matrix[0].length}; }
    
    Double[][] resize(Double[][] matrix, int[] sizes) {
     
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
    
    Double[] matrix2vec(Double[][] matrix) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[] out = new Double[matrix.length * matrix[0].length];
        
        int ctr = -1;
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++) {
              
                ctr++;
                out[ctr] = matrix[i][j];
              
            }
        
         return out;
      
    }
    
    Double[][] vector2matrix(Double[] vector, int[] sizes) {

        Double[][] out = new Double[sizes[0]][sizes[1]];
        
        int ctr = -1;
        for (int i = 0; i < sizes[0]; i++)
            for (int j = 0; j < sizes[1]; j++) {
              
                ctr++;
                out[i][j] = vector[ctr];
              
            }
        
         return out;
      
    }
    
    Double[][] transpose(Double[][] matrix) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_cols][hm_rows];
                
        for (int i = 0; i < hm_cols; i++)
            for (int j = 0; j < hm_rows; j++)
                out[i][j] = matrix[j][i];
                
        return out;
        
    }

    double sum(Double[][] matrix) {

        double sum = 0;
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                sum += matrix[i][j];


        return sum;

    }
    
    // special operations

    Double[][] exp(Double[][] matrix, double power) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = Math.pow(matrix[i][j], power);
                
        return out;
        
    }

    Double[][] sigm(Double[][] matrix) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = (1.0 / (1 + Math.exp(-matrix[i][j])));

        return out;
       
    }
    
    Double[][] tanh(Double[][] matrix) {
      
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = Math.tanh(matrix[i][j]);

        return out;
       
    }

    Double[][] cross_entropy(Double[][] target, Double[][] output) {

        int hm_rows = target.length;
        int hm_cols = target[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
      
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = (target[i][j] * Math.log(output[i][j])) + ((1 - target[i][j]) * Math.log(1 - output[i][j]));

        return out;
   
    }
    
    Double[][] softmax(Double[][] matrix) {
        
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
        
        double sum = 0;
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                sum += Math.exp(matrix[i][j]);
      
        if (hm_rows == 1)
            for (int k = 0; k < hm_cols; k++)
                out[0][k] = Math.exp(matrix[0][k])/sum;
        
        if (hm_cols == 1)
            for (int k = 0; k < hm_rows; k++)
                out[k][0] = Math.exp(matrix[k][0])/sum;
        
        return out;
        
    }
    
    Double[][] softmax_partional(Double[][] matrix, int index_begin, int index_end) {
        
        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
        
        double sum = 0;
      
        if (hm_rows == 1) {
          
            for (int k = index_begin; k < index_end; k++)
                sum += Math.exp(matrix[0][k]);
          
            for (int k = index_begin; k < index_end; k++)
                out[0][k] = Math.exp(matrix[0][k])/sum;
           
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
