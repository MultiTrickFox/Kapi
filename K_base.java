import java.util.Arrays;
import java.util.Random;

// 2DO: compile as .jar

class K_base {

    Random random;
    
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
    
    int[] size(Double[][] a) { return new int[]{a.length,a[0].length}; }
    
    Double[][] resize(Double[][] a, int[] b) {
     
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[b[0]][b[1]];
        
        int ctr = -1;
        for (int i = 0; i < b[0]; i++)
            for (int j = 0; j < b[1]; j++) {
              
                ctr++;
                // out[i][j] = a[][]; // TODO : derive row/col here.
              
            }
       
         return out;
       
    }
    
    Double[] matrix2vec(Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[] out = new Double[a.length * a[0].length];
        
        int ctr = -1;
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++) {
              
                ctr++;
                out[ctr] = a[i][j];
              
            }
        
         return out;
      
    }
    
    Double[][] vec2matrix(Double[] a, int[] b) {

        Double[][] out = new Double[b[0]][b[1]];
        
        int ctr = -1;
        for (int i = 0; i < b[0]; i++)
            for (int j = 0; j < b[1]; j++) {
              
                ctr++;
                out[i][j] = a[ctr];
              
            }
        
         return out;
      
    }
    
    Double[][] transpose(Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_cols][hm_rows];
                
        for (int i = 0; i < hm_cols; i++)
            for (int j = 0; j < hm_rows; j++)
                out[i][j] = a[j][i];
                
        return out;
        
    }
    
    // special operations

    Double[][] exp(Double[][] a, double e) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = Math.pow(a[i][j], e);
                
        return out;
        
    }

    Double[][] sigm(Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = (1.0 / (1 + Math.exp(-a[i][j])));

        return out;
       
    }
    
    Double[][] tanh(Double[][] a) {
      
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = Math.tanh(a[i][j]);

        return out;
       
    }

    Double[][] cross_entropy(Double[][] target, Double[][] label) {

        int hm_rows = target.length;
        int hm_cols = target[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
      
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                out[i][j] = (target[i][j] * Math.log(label[i][j])) + ((1 - target[i][j]) * Math.log(1 - label[i][j]));

        return out;
   
    }
    
    Double[][] softmax(Double[][] a) {
        
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
        
        double sum = 0;
        for (int i = 0; i < hm_rows; i++)
            for (int j = 0; j < hm_cols; j++)
                sum += Math.exp(a[i][j]);
      
        if (hm_rows == 1)
            for (int k = 0; k < hm_cols; k++)
                out[0][k] = Math.exp(a[0][k])/sum;
        
        if (hm_cols == 1)
            for (int k = 0; k < hm_rows; k++)
                out[k][0] = Math.exp(a[k][0])/sum;
        
        return out;
        
    }
    
    Double[][] softmax_partional(Double[][] a, int index_begin, int index_end) {
        
        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Double[][] out = new Double[hm_rows][hm_cols];
        
        double sum = 0;
      
        if (hm_rows == 1) {
          
            for (int k = index_begin; k < index_end; k++)
                sum += Math.exp(a[0][k]);
          
            for (int k = index_begin; k < index_end; k++)
                out[0][k] = Math.exp(a[0][k])/sum;
           
        }
            
        if (hm_cols == 1) {
          
            for (int k = index_begin; k < index_end; k++)
                sum += Math.exp(a[k][0]); 
          
            for (int k = index_begin; k < index_end; k++)
                out[k][0] = Math.exp(a[k][0])/sum;
          
        }
            
        return out;
        
    }

}
