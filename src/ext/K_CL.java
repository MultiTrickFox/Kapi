package ext;

import java.util.HashMap;

import static org.jocl.CL.*;
import org.jocl.*;


class K_CL {


    public static class K_Bindings {

        int platformIndex;
        long deviceType = CL_DEVICE_TYPE_ALL;
        int deviceIndex;

        cl_context_properties contextProperties;
        cl_context context;
        cl_command_queue commandQueue;

        HashMap<String,cl_kernel> kernels;


        public void init() {

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
            cl_device_id device = devices[deviceIndex];


            // Initialize the context properties
            contextProperties = new cl_context_properties();
            contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
            // Create a context for the selected device
            context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
            // Create a command-queue for the selected device
            commandQueue = clCreateCommandQueue(context, device, 0, null);

            System.out.println("num devices: " + numDevices); // todo: show extra details,names,etc here.


            // for matmul..

            // Create the program from the source code
            cl_program program_matmul = clCreateProgramWithSource(context,1, new String[]{ K_Kernels.matmul }, null, null);
            // Build the program
            clBuildProgram(program_matmul, 0, null, null, null, null);
            // Create the kernel
            cl_kernel kernel_matmul = clCreateKernel(program_matmul, "matmul", null);
            kernels.put("matmul",kernel_matmul);




        }


        private static class K_Kernels {

            static String matmul =
                    "__kernel void matrixMul(" +
                            "          __global float* C, " +
                            "          __global float* A, " +
                            "          __global float* B, " +
                            "          int wA, int wB)" +
                            "{ " +
                            " "+
                            "   int tx = get_global_id(0); " +
                            "   int ty = get_global_id(1); " +
                            " " +
                            "   // value stores the element that is " +
                            "   // computed by the thread" +
                            "   float value = 0;" +
                            "   for (int k = 0; k < wA; ++k)" +
                            "   {" +
                            "      float elementA = A[ty * wA + k];" +
                            "      float elementB = B[k * wB + tx];" +
                            "      value += elementA * elementB;" +
                            "   }" +
                            " " +
                            "   // Write the matrix to device memory each " +
                            "   // thread writes one element" +
                            "   C[ty * wA + tx] = value;" +
                            " "+
                            "}";

        }




















    }

}


