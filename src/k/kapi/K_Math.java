package k.kapi;

import java.util.Random;


class K_Math {


    static Random random = new Random(System.currentTimeMillis());

    static float mean = 0;
    static float dev = 1;


    // constructor

    K_Math() { } static K_Math instance = new K_Math();


    // matrix initializers

    static Float[][] zeros(int hm_rows, int hm_cols) {

        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = 0.0f;
        }

        return out;

    }

    static Float[][] ones(int hm_rows, int hm_cols) {

        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = 1.0f;
        }

        return out;

    }

    static Float[][] randn(int hm_rows, int hm_cols) {

        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = (float) random.nextGaussian() * dev + mean;
        }

        return out;

    }

    static Float[][] identity(int hm_rows, int hm_cols) {

        assert hm_rows == hm_cols;

        Float[][] out = new Float[hm_rows][hm_cols];

        for (int i = 0; i < hm_rows; i++)
            out[i][i] = 1.0f;

        return out;

    } static Float[][] identity(int hm_rowcols) { return identity(hm_rowcols, hm_rowcols); }

    static Float[][] constants(int hm_rows, int hm_cols, float value) {

        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = value;
        }

        return out;

    }


    // matrix operations

    static Float[][] add(Float[][] a, Float[][] b) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a, row_b;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            row_b = b[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] + row_b[j];
        }

        return out;

    }

    static Float[][] sub(Float[][] a, Float[][] b) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a, row_b;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            row_b = b[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] - row_b[j];
        }

        return out;

    }

    static Float[][] mul(Float[][] a, Float[][] b) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a, row_b;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            row_b = b[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] * row_b[j];
        }

        return out;

    }

    static Float[][] div(Float[][] a, Float[][] b) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a, row_b;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            row_b = b[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] / row_b[j];
        }

        return out;

    }

    static Float[][] matmul(Float[][] a, Float[][] b) {

        int hm_rows1 = a.length;
        int hm_cols1 = a[0].length;
        int hm_rows2 = b.length;
        int hm_cols2 = b[0].length;
        Float[][] out = new Float[hm_rows1][hm_cols2];

        assert hm_cols1 == hm_rows2;

        Float[] row, row_a;

        for (int i = 0; i < hm_rows1; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols2; j++) {
                row[j] = 0.0f;
                for (int k = 0; k < hm_cols1; k++)
                    row[j] += row_a[k] * b[k][j];
            }
        }

        return out;

    }


    // scalar operations

    static Float[][] mul_scalar(float b, Float[][] a) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] * b;
        }

        return out;

    } static Float[][] mul_scalar(Float[][] a, float b) { return mul_scalar(b, a); }

    static Float[][] div_scalar(float b, Float[][] a) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = b / row_a[j];
        }

        return out;

    } static Float[][] div_scalar(Float[][] a, float b) { return mul_scalar(1/b, a); }

    static Float[][] sub_scalar(float b, Float[][] a) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = b - row_a[j];
        }

        return out;

    }

    static Float[][] sub_scalar(Float[][] a, float b) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] - b;
        }

        return out;

    }

    static Float[][] add_scalar(float b, Float[][] a) {

        int hm_rows = a.length;
        int hm_cols = a[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = a[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = row_a[j] + b;
        }

        return out;

    } static Float[][] add_scalar(Float[][] a, float b) { return add_scalar(b, a); }


    // helpers

    static int size(Float[][] matrix, int dim) {

        if (dim == 0) return matrix.length;
        else          return matrix[0].length;

    } static int[] size(Float[][] matrix) { return new int[]{matrix.length,matrix[0].length}; }

    static Float[][] resize(Float[][] matrix, int[] sizes) {

        return vector2matrix(matrix2vector(matrix), sizes);

    }

    static Float[][] resize(Float[][] matrix, int size1, int size2) {

        return resize(matrix, new int[]{size1, size2});

    }

    static Float[] matrix2vector(Float[][] matrix) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[] out = new Float[matrix.length * matrix[0].length];

        int ctr = -1;
        for (Float[] row : matrix) {
            for (int j = 0; j < hm_cols; j++) {

                ctr++;
                out[ctr] = row[j];

            }
        }

        return out;

    }

    static Float[][] vector2matrix(Float[] vector, int[] sizes) {

        Float[][] out = new Float[sizes[0]][sizes[1]];

        int ctr = -1;
        for (Float[] row : out) {
            for (int j = 0; j < sizes[1]; j++) {

                ctr++;
                row[j] = vector[ctr];

            }
        }

        return out;

    } static Float[][] vector2matrix(Float[] vector, int size1, int size2) { return vector2matrix(vector, new int[]{size1, size2}); }

    static Float[][] transpose(Float[][] matrix) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = new Float[hm_cols][hm_rows];

        Float[] row;

        for (int i = 0; i < matrix.length; i++) {
            row = matrix[i];
            for (int j = 0; j < matrix[0].length; j++)
                out[j][i] = row[j];
        }

        return out;

    }

    static float sum(Float[][] matrix) {

        float sum = 0;

        for (Float[] row : matrix)
            for (Float col : row)
                sum += col;

        return sum;

    }

    static Float[][] sum(Float[][] matrix, int dim) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = null;

        if (dim == 0) {

            out = new Float[1][hm_cols];
            Float[] row = out[0];
            float col_sum;

            for (int j = 0; j < hm_cols; j++) {
                col_sum = 0;

                for (int i = 0; i < hm_rows; i++)
                    col_sum += matrix[i][j];
                row[j] = col_sum;

            }

        }

        if (dim == 1) {

            out = new Float[hm_rows][1];
            float row_sum;

            int i = -1;
            for (Float[] row : matrix) {
                i++;
                row_sum = 0;

                for (Float e : row)
                    row_sum += e;
                out[i][0] = row_sum;

            }

        }

        return out;

    }


    // vector operations

    static Float[] vector_mul(Float[] v1, Float[] v2) {

        assert v1.length == v2.length;

        int hm_elements = v1.length;
        Float[] out = new Float[hm_elements];

        for (int i = 0; i < hm_elements; i++)
            out[i] = v1[i] * v2[i];

        return out;

    }

    static Float[] vector_add(Float[] v1, Float[] v2) {

        assert v1.length == v2.length;

        int hm_elements = v1.length;
        Float[] out = new Float[hm_elements];

        for (int i = 0; i < hm_elements; i++)
            out[i] = v1[i] + v2[i];

        return out;

    }

    static float vector_sum(Float[] v) {

        float doubl = 0;

        for (Float d : v) doubl += d;

        return doubl;

    }


    // special operations

    static Float[][] pow(Float[][] matrix, float power) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = (float) Math.pow(row_a[j], power);
        }

        return out;

    }

    static Float[][] exp(Float[][] matrix) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = (float) Math.exp(row_a[j]);
        }

        return out;

    }

    static Float[][] log(Float[][] matrix) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = (float) Math.log(row_a[j]);
        }

        return out;

    }

    static Float[][] sigm(Float[][] matrix) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = (float) (1.0 / (1 + Math.exp(-row_a[j])));
        }

        return out;

    }

    static Float[][] tanh(Float[][] matrix) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = (float) Math.tanh(row_a[j]);
        }

        return out;

    }

    static Float[][] cross_entropy(Float[][] target, Float[][] output) {

        int hm_rows = target.length;
        int hm_cols = target[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a, row_b;

        for (int i = 0; i < hm_rows; i++) {
            row = out[i];
            row_a = target[i];
            row_b = output[i];
            for (int j = 0; j < hm_cols; j++)
                row[j] = (float) -(row_a[j]*Math.log(row_b[j]));
        }

        return out;

    }

    static Float[][] softmax(Float[][] matrix) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        Float[] row, row_a;

        float sum = 0;

        for (int i = 0; i < hm_rows; i++) {
            row_a = matrix[i];
            for (int j = 0; j < hm_cols; j++)
                sum += Math.exp(row_a[j]);
        }

        if (hm_rows == 1) {

            row = out[0];
            row_a = matrix[0];

            for (int k = 0; k < hm_cols; k++)
                row[k] = (float) Math.exp(row_a[k]) / sum;
        }

        if (hm_cols == 1)

            for (int k = 0; k < hm_rows; k++)
                out[k][0] = (float) Math.exp(matrix[k][0])/sum;

        return out;

    }

    static Float[][] softmax(Float[][] matrix, int index_begin, int index_end) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        Float[][] out = new Float[hm_rows][hm_cols];

        float sum = 0;

        if (hm_rows == 1) {

            Float[] row = out[0];
            Float[] row_a = matrix[0];

            for (int k = index_begin; k < index_end; k++)
                sum += Math.exp(row_a[k]);
            for (int k = index_begin; k < index_end; k++)
                row[k] = (float) Math.exp(row_a[k])/sum;

        }

        if (hm_cols == 1) {

            for (int k = index_begin; k < index_end; k++)
                sum += Math.exp(matrix[k][0]);
            for (int k = index_begin; k < index_end; k++)
                out[k][0] = (float) Math.exp(matrix[k][0])/sum;

        }

        return out;

    }


}