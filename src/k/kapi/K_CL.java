package k.kapi;

import static org.jocl.CL.*;
import org.jocl.*;

import java.util.HashMap;


class K_CL {


    static int platformIndex;
    static long deviceType = CL_DEVICE_TYPE_ALL;
    static int deviceIndex;

    static cl_device_id device;
    static cl_context_properties contextProperties;
    static cl_context context;
    static cl_command_queue commandQueue;

    static HashMap<String,cl_program> programs;

    static boolean gpu_enabled = false;

    final static Object lock = new Object();


    static void init() {

        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
        platformIndex = numPlatforms-1;
        // Obtain a platform ID
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        deviceIndex = numDevices-1;
        // Obtain a device ID
        cl_device_id[] devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        device = devices[deviceIndex];

        // Initialize the context properties
        contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        // Create a context for the selected device
        context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
        // Create a command-queue for the selected device
        commandQueue = clCreateCommandQueue(context, device, 0, null);

        switch(deviceIndex) {

            case 0:
                System.out.println("K_CL device_index: {0} - no gpu available.");
                gpu_enabled = false;
                break;
            case 1:
                System.out.println("K_CL device_index: {1} - internal gpu will be used.");
                gpu_enabled = true;
                break;
            default:
                System.out.println("K_CL device_index: {" + deviceIndex + "} - external gpu will be used.");
                gpu_enabled = true;

        }

        programs = new HashMap<>();

        cl_program program_matmul = clCreateProgramWithSource(context,1, new String[]{ Kernels.matmul }, null, null);
        clBuildProgram(program_matmul, 0, null, null, null, null);
        programs.put("matmul",program_matmul);

        cl_program program_mul = clCreateProgramWithSource(context,1, new String[]{ Kernels.mul }, null, null);
        clBuildProgram(program_mul, 0, null, null, null, null);
        programs.put("mul",program_mul);

        cl_program program_add = clCreateProgramWithSource(context,1, new String[]{ Kernels.add }, null, null);
        clBuildProgram(program_add, 0, null, null, null, null);
        programs.put("add",program_add);

        cl_program program_sub = clCreateProgramWithSource(context,1, new String[]{ Kernels.sub }, null, null);
        clBuildProgram(program_sub, 0, null, null, null, null);
        programs.put("sub",program_sub);
        
    }


    static void abort() {

        for (cl_program program : programs.values())
            clReleaseProgram(program);

        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        clReleaseDevice(device);

    }


    static class OPS {


        static Float[][] matmul(Float[][] A, Float[][] B) {

            cl_kernel kernel = clCreateKernel(programs.get("matmul"), "matmul", null);
            
            float[] a = matrix2vector(A);
            float[] b = matrix2vector(B);
            float[] c = new float[A.length * B[0].length];
            float[] args = new float[]{A.length, A[0].length, B.length, B[0].length};

            Pointer srcA = Pointer.to(a);
            Pointer srcB = Pointer.to(b);
            Pointer dst = Pointer.to(c);
            Pointer ext = Pointer.to(args);

            cl_mem[] memObjects = new cl_mem[4];
            memObjects[0] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * a.length, srcA, null);
            memObjects[1] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * b.length, srcB, null);
            memObjects[2] = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    Sizeof.cl_float * c.length, null, null);
            memObjects[3] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * args.length, ext, null);

            // synchronized (lock) {

                clSetKernelArg(kernel, 0,
                        Sizeof.cl_mem, Pointer.to(memObjects[0]));
                clSetKernelArg(kernel, 1,
                        Sizeof.cl_mem, Pointer.to(memObjects[1]));
                clSetKernelArg(kernel, 2,
                        Sizeof.cl_mem, Pointer.to(memObjects[2]));
                clSetKernelArg(kernel, 3,
                        Sizeof.cl_mem, Pointer.to(memObjects[3]));

                long[] global_work_size = new long[]{c.length};
                long[] local_work_size = new long[]{1};

                clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                        global_work_size, local_work_size, 0, null, null);

                clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                        c.length * Sizeof.cl_float, dst, 0, null, null);

                clReleaseMemObject(memObjects[0]);
                clReleaseMemObject(memObjects[1]);
                clReleaseMemObject(memObjects[2]);
                clReleaseMemObject(memObjects[3]);

                clReleaseKernel(kernel);

                return vector2matrix(c, A.length, B[0].length);

            // }

        }

        static Float[][] mul(Float[][] A, Float[][] B) {

            cl_kernel kernel = clCreateKernel(programs.get("mul"), "mul", null);

            float[] a = matrix2vector(A);
            float[] b = matrix2vector(B);
            float[] c = new float[A.length * A[0].length];

            Pointer srcA = Pointer.to(a);
            Pointer srcB = Pointer.to(b);
            Pointer dst = Pointer.to(c);

            cl_mem[] memObjects = new cl_mem[3];
            memObjects[0] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * a.length, srcA, null);
            memObjects[1] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * b.length, srcB, null);
            memObjects[2] = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    Sizeof.cl_float * c.length, null, null);

            clSetKernelArg(kernel, 0,
                    Sizeof.cl_mem, Pointer.to(memObjects[0]));
            clSetKernelArg(kernel, 1,
                    Sizeof.cl_mem, Pointer.to(memObjects[1]));
            clSetKernelArg(kernel, 2,
                    Sizeof.cl_mem, Pointer.to(memObjects[2]));

            long[] global_work_size = new long[]{c.length};
            long[] local_work_size = new long[]{1};

            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                    global_work_size, local_work_size, 0, null, null);

            clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                    c.length * Sizeof.cl_float, dst, 0, null, null);

            clReleaseMemObject(memObjects[0]);
            clReleaseMemObject(memObjects[1]);
            clReleaseMemObject(memObjects[2]);

            clReleaseKernel(kernel);

            return vector2matrix(c, A.length, A[0].length);

        }

        static Float[] mul(Float[] A, Float[] B) {

            cl_kernel kernel = clCreateKernel(programs.get("mul"), "mul", null);

            float[] a = new float[A.length];
            float[] b = new float[B.length];
            float[] c = new float[A.length];

            int ctr;

            ctr = -1;
            for (Float f : A) {
                ctr++;
                a[ctr] = f;
            }

            ctr = -1;
            for (Float f : B) {
                ctr++;
                b[ctr] = f;
            }

            Pointer srcA = Pointer.to(a);
            Pointer srcB = Pointer.to(b);
            Pointer dst = Pointer.to(c);

            cl_mem[] memObjects = new cl_mem[3];
            memObjects[0] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * a.length, srcA, null);
            memObjects[1] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * b.length, srcB, null);
            memObjects[2] = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    Sizeof.cl_float * c.length, null, null);

            clSetKernelArg(kernel, 0,
                    Sizeof.cl_mem, Pointer.to(memObjects[0]));
            clSetKernelArg(kernel, 1,
                    Sizeof.cl_mem, Pointer.to(memObjects[1]));
            clSetKernelArg(kernel, 2,
                    Sizeof.cl_mem, Pointer.to(memObjects[2]));

            long[] global_work_size = new long[]{c.length};
            long[] local_work_size = new long[]{1};

            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                    global_work_size, local_work_size, 0, null, null);

            clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                    c.length * Sizeof.cl_float, dst, 0, null, null);

            clReleaseMemObject(memObjects[0]);
            clReleaseMemObject(memObjects[1]);
            clReleaseMemObject(memObjects[2]);

            clReleaseKernel(kernel);

            Float[] C  = new Float[c.length];

            ctr = -1;
            for (Float f : c) {
                ctr++;
                C[ctr] = f;
            }

            return C;

        }

        static Float[][] add(Float[][] A, Float[][] B) {

            cl_kernel kernel = clCreateKernel(programs.get("add"), "add", null);

            float[] a = matrix2vector(A);
            float[] b = matrix2vector(B);
            float[] c = new float[A.length * A[0].length];

            Pointer srcA = Pointer.to(a);
            Pointer srcB = Pointer.to(b);
            Pointer dst = Pointer.to(c);

            cl_mem[] memObjects = new cl_mem[3];
            memObjects[0] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * a.length, srcA, null);
            memObjects[1] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * b.length, srcB, null);
            memObjects[2] = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    Sizeof.cl_float * c.length, null, null);

            clSetKernelArg(kernel, 0,
                    Sizeof.cl_mem, Pointer.to(memObjects[0]));
            clSetKernelArg(kernel, 1,
                    Sizeof.cl_mem, Pointer.to(memObjects[1]));
            clSetKernelArg(kernel, 2,
                    Sizeof.cl_mem, Pointer.to(memObjects[2]));

            long[] global_work_size = new long[]{c.length};
            long[] local_work_size = new long[]{1};

            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                    global_work_size, local_work_size, 0, null, null);

            clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                    c.length * Sizeof.cl_float, dst, 0, null, null);

            clReleaseMemObject(memObjects[0]);
            clReleaseMemObject(memObjects[1]);
            clReleaseMemObject(memObjects[2]);

            clReleaseKernel(kernel);

            return vector2matrix(c, A.length, A[0].length);

        }

        static Float[] add(Float[] A, Float[] B) {

            cl_kernel kernel = clCreateKernel(programs.get("add"), "add", null);

            float[] a = new float[A.length];
            float[] b = new float[B.length];
            float[] c = new float[A.length];

            int ctr;

            ctr = -1;
            for (Float f : A) {
                ctr++;
                a[ctr] = f;
            }

            ctr = -1;
            for (Float f : B) {
                ctr++;
                b[ctr] = f;
            }

            Pointer srcA = Pointer.to(a);
            Pointer srcB = Pointer.to(b);
            Pointer dst = Pointer.to(c);

            cl_mem[] memObjects = new cl_mem[3];
            memObjects[0] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * a.length, srcA, null);
            memObjects[1] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * b.length, srcB, null);
            memObjects[2] = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    Sizeof.cl_float * c.length, null, null);

            clSetKernelArg(kernel, 0,
                    Sizeof.cl_mem, Pointer.to(memObjects[0]));
            clSetKernelArg(kernel, 1,
                    Sizeof.cl_mem, Pointer.to(memObjects[1]));
            clSetKernelArg(kernel, 2,
                    Sizeof.cl_mem, Pointer.to(memObjects[2]));

            long[] global_work_size = new long[]{c.length};
            long[] local_work_size = new long[]{1};

            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                    global_work_size, local_work_size, 0, null, null);

            clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                    c.length * Sizeof.cl_float, dst, 0, null, null);

            clReleaseMemObject(memObjects[0]);
            clReleaseMemObject(memObjects[1]);
            clReleaseMemObject(memObjects[2]);

            clReleaseKernel(kernel);

            Float[] C  = new Float[c.length];

            ctr = -1;
            for (Float f : c) {
                ctr++;
                C[ctr] = f;
            }

            return C;

        }

        static Float[][] sub(Float[][] A, Float[][] B) {

            cl_kernel kernel = clCreateKernel(programs.get("sub"), "sub", null);

            float[] a = matrix2vector(A);
            float[] b = matrix2vector(B);
            float[] c = new float[A.length * A[0].length];

            Pointer srcA = Pointer.to(a);
            Pointer srcB = Pointer.to(b);
            Pointer dst = Pointer.to(c);

            cl_mem[] memObjects = new cl_mem[3];
            memObjects[0] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * a.length, srcA, null);
            memObjects[1] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * b.length, srcB, null);
            memObjects[2] = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    Sizeof.cl_float * c.length, null, null);

            clSetKernelArg(kernel, 0,
                    Sizeof.cl_mem, Pointer.to(memObjects[0]));
            clSetKernelArg(kernel, 1,
                    Sizeof.cl_mem, Pointer.to(memObjects[1]));
            clSetKernelArg(kernel, 2,
                    Sizeof.cl_mem, Pointer.to(memObjects[2]));

            long[] global_work_size = new long[]{c.length};
            long[] local_work_size = new long[]{1};

            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                    global_work_size, local_work_size, 0, null, null);

            clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                    c.length * Sizeof.cl_float, dst, 0, null, null);

            clReleaseMemObject(memObjects[0]);
            clReleaseMemObject(memObjects[1]);
            clReleaseMemObject(memObjects[2]);

            clReleaseKernel(kernel);

            return vector2matrix(c, A.length, A[0].length);

        }

        static Float[] sub(Float[] A, Float[] B) {

            cl_kernel kernel = clCreateKernel(programs.get("sub"), "sub", null);

            float[] a = new float[A.length];
            float[] b = new float[B.length];
            float[] c = new float[A.length];

            int ctr;

            ctr = -1;
            for (Float f : A) {
                ctr++;
                a[ctr] = f;
            }

            ctr = -1;
            for (Float f : B) {
                ctr++;
                b[ctr] = f;
            }

            Pointer srcA = Pointer.to(a);
            Pointer srcB = Pointer.to(b);
            Pointer dst = Pointer.to(c);

            cl_mem[] memObjects = new cl_mem[3];
            memObjects[0] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * a.length, srcA, null);
            memObjects[1] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * b.length, srcB, null);
            memObjects[2] = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    Sizeof.cl_float * c.length, null, null);

            clSetKernelArg(kernel, 0,
                    Sizeof.cl_mem, Pointer.to(memObjects[0]));
            clSetKernelArg(kernel, 1,
                    Sizeof.cl_mem, Pointer.to(memObjects[1]));
            clSetKernelArg(kernel, 2,
                    Sizeof.cl_mem, Pointer.to(memObjects[2]));

            long[] global_work_size = new long[]{c.length};
            long[] local_work_size = new long[]{1};

            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                    global_work_size, local_work_size, 0, null, null);

            clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                    c.length * Sizeof.cl_float, dst, 0, null, null);

            clReleaseMemObject(memObjects[0]);
            clReleaseMemObject(memObjects[1]);
            clReleaseMemObject(memObjects[2]);

            clReleaseKernel(kernel);

            Float[] C  = new Float[c.length];

            ctr = -1;
            for (Float f : c) {
                ctr++;
                C[ctr] = f;
            }

            return C;

        }


    }


    private static class Kernels {


        static String matmul =
                "__kernel void matmul(" +
                        "" +
                        "__global const float *a," +
                        "__global const float *b," +
                        "__global       float *c," +
                        "__global const float *ext)" +
                        "{" +
                        "        int hm_rows1 = ext[0];" +
                        "        int hm_cols1 = ext[1];" +
                        "        int hm_rows2 = ext[2];" +
                        "        int hm_cols2 = ext[3];" +
                        "        int gid = get_global_id(0);" +
                        "        int col = gid%hm_cols2;" +
                        "        int row = gid/hm_cols2;" +
                        "" +
                        "        float sum = 0;" +
                        "        for (int j = 0; j < hm_cols1; j++) {" +
                        "             sum += a[row*hm_cols1+j] * b[j*hm_cols2+col];" +
                        "        }" +
                        "        c[gid] = sum;" +
                        "}";

        static String mul =
                "__kernel void mul(" +
                        "" +
                        "__global const float *a," +
                        "__global const float *b," +
                        "__global       float *c" +
                        "){" +
                        "        int gid = get_global_id(0);" +
                        "        c[gid] = a[gid] * b[gid];" +
                        "}";

        static String add =
                "__kernel void add(" +
                        "" +
                        "__global const float *a," +
                        "__global const float *b," +
                        "__global       float *c" +
                        "){" +
                        "        int gid = get_global_id(0);" +
                        "        c[gid] = a[gid] + b[gid];" +
                        "}";

        static String sub =
                "__kernel void sub(" +
                        "" +
                        "__global const float *a," +
                        "__global const float *b," +
                        "__global       float *c" +
                        "){" +
                        "        int gid = get_global_id(0);" +
                        "        c[gid] = a[gid] - b[gid];" +
                        "}";


    }


    static float[] matrix2vector(Float[][] matrix) {

        int hm_rows = matrix.length;
        int hm_cols = matrix[0].length;
        float[] out = new float[matrix.length * matrix[0].length];

        int ctr = -1;
        for (Float[] row : matrix) {
            for (int j = 0; j < hm_cols; j++) {

                ctr++;
                out[ctr] = row[j];

            }
        }

        return out;

    }

    static Float[][] vector2matrix(float[] vector, int[] sizes) {

        Float[][] out = new Float[sizes[0]][sizes[1]];

        int ctr = -1;
        for (Float[] row : out) {
            for (int j = 0; j < sizes[1]; j++) {

                ctr++;
                row[j] = vector[ctr];

            }
        }

        return out;

    } static Float[][] vector2matrix(float[] vector, int size1, int size2) { return vector2matrix(vector, new int[]{size1, size2}); }


}


