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
    static HashMap<String,cl_kernel> kernels;

    static boolean gpu_enabled;

    private final static Object lock = new Object();


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
                System.out.println("K_CL: device_index {0} - cpu will NOT be used.");
                gpu_enabled = false;
                break;
            case 1:
                System.out.println("K_CL: device_index {1} - internal gpu will be used.");
                gpu_enabled = true;
                break;
            default:
                System.out.println("K_CL: device_index {" + deviceIndex + "} - external gpu will be used.");
                gpu_enabled = true;
                                                                            // todo: show extra details,names,etc here.
        }


        programs = new HashMap<>();
        kernels = new HashMap<>();


        // for matmul

        // Create the program from the source code
        cl_program program_matmul = clCreateProgramWithSource(context,1, new String[]{ Kernels.matmul }, null, null);
        // Build the program
        clBuildProgram(program_matmul, 0, null, null, null, null);
        programs.put("matmul",program_matmul);
        // Create the kernel
        cl_kernel kernel_matmul = clCreateKernel(program_matmul, "matmul", null);
        kernels.put("matmul",kernel_matmul);


    }


    static void abort() {

        for (cl_kernel kernel : kernels.values())
            clReleaseKernel(kernel);

        for (cl_program program : programs.values())
            clReleaseProgram(program);

        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        clReleaseDevice(device);

    }


    static class OPS {


        static Float[][] matmul(Float[][] A, Float[][] B) {

            cl_kernel kernel = clCreateKernel(programs.get("matmul"), "matmul", null);

            // cl_kernel kernel = kernels.get("matmul"); // TODO: find how to make every thread use the same kernel OR is it possible?

            float[] a = new float[K_Math.size(A, 0) * K_Math.size(A, 1)];
            float[] b = new float[K_Math.size(B, 0) * K_Math.size(B, 1)];
            float[] c = new float[A.length * B[0].length];

            int ctr;

            ctr = -1;
            for (Float f : K_Math.matrix2vector(A)) { // TODO : just use your own float[][] matrix2vector here
                ctr++;
                a[ctr] = f;
            }

            ctr = -1;
            for (Float f : K_Math.matrix2vector(B)) {
                ctr++;
                b[ctr] = f;
            }

            float[] args = new float[]{A.length, A[0].length, B.length, B[0].length};

            Pointer srcA = Pointer.to(a);
            Pointer srcB = Pointer.to(b);
            Pointer dst = Pointer.to(c);
            Pointer ext = Pointer.to(args);

            // Allocate the memory objects for the input- and output data
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



            //synchronized (lock) {

                // Set the arguments for the kernel
                clSetKernelArg(kernel, 0,
                        Sizeof.cl_mem, Pointer.to(memObjects[0]));
                clSetKernelArg(kernel, 1,
                        Sizeof.cl_mem, Pointer.to(memObjects[1]));
                clSetKernelArg(kernel, 2,
                        Sizeof.cl_mem, Pointer.to(memObjects[2]));
                clSetKernelArg(kernel, 3,
                        Sizeof.cl_mem, Pointer.to(memObjects[3]));

                // Set the work-item dimensions
                long[] global_work_size = new long[]{c.length};
                long[] local_work_size = new long[]{1};

                // Execute the kernel
                clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                        global_work_size, local_work_size, 0, null, null);

                // Read the output data
                clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                        c.length * Sizeof.cl_float, dst, 0, null, null);

                // Release memory

                clReleaseMemObject(memObjects[0]);
                clReleaseMemObject(memObjects[1]);
                clReleaseMemObject(memObjects[2]);
                clReleaseMemObject(memObjects[3]);

                Float[] C = new Float[c.length];

//                for (float f : c) { // TODO : remove me.
//
//                    System.out.println(f);
//
//                }

                ctr = -1;
                for (float f : c) {
                    ctr++;
                    C[ctr] = f;
                }

                clReleaseKernel(kernel);

                return K_Math.vector2matrix(C, A.length, B[0].length); // TODO : just use your own float[][] vector2matrix here

            //}

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
                        //"            for (int i = 0; i < hm_rows2; i++) {" +
                        "                sum += a[row*hm_cols1+j] * b[j*hm_cols2+col];" +
                        //"            }" +
                        "        }" +
                        "        c[gid] = sum;" +
                        "}";

//        private static String matmul_bwA = "";
//
//        private static String matmul_bwB = "";





//        private static String matmul_old =
//
//                "__kernel void matmul(" +
//
//                        " __global const float* a, " +
//                        " __global const float* b, " +
//                        " __global       float* c, " +
//                        " __global const float* sizes" +
//
//                        ") { " +
//                        " " +
//
//                        "   int rows_a = sizes[0]; " +
//                        "   int cols_a = sizes[1]; " +
//                        "   int rows_b = sizes[2]; " +
//                        "   int cols_b = sizes[3]; " +
//                        "   int x = get_global_id(0); " +
//                        "   int y = get_global_id(1); " +
//                        "   int ctr = 0; " +
//                        "   for (int i = 0; i < cols_a; i++) {" +
//                        "       ctr +=                            "
//                        "" +
//                        "" +
//                        "" +
//                        ""
//
//
//                        " "+
//                        "}";
//
//
//
//                        //"          int wA, int wB)" +
//
////                        " " +
////                        "   // value stores the element that is " +
////                        "   // computed by the thread" +
////                        "   float value = 0;" +
////                        "   for (int k = 0; k < wA; ++k)" +
////                        "   {" +
////                        //"      float elementA = A[ty * wA + k];" +
////                        //"      float elementB = B[k * wB + tx];" +
////                        //"      value += elementA * elementB;" +
////                        "   }" +
////                        " " +
////                        "   // Write the matrix to device memory each " +
////                        "   // thread writes one element" +
////                        //"   C[ty * wA + tx] = value;" +
////                        " "+
//
    }


}


