#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "matvec.cl"
#define KERNEL_FUNC "matvec_mult"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void imprime_matriz(int *matriz, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%d ", matriz[j + width * i]);
		}
		printf("\n");
	}
}

int main() {

	int GRID_SIZE = 1024;

	/* Host/device data structures */
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_int i, err;

	/* Program/kernel data structures */
	cl_program program;
	FILE * program_handle;
	char * program_buffer, *program_log;
	size_t program_size, log_size;
	cl_kernel kernel;

	/* Data and buffers */
	int *matriza;
	int *matrizb;
	int *matrizc;
	cl_mem mat_buff, vec_buff, res_buff;
	size_t work_units_per_kernel;

	if ((matriza = (int *) calloc((GRID_SIZE), sizeof(int))) == NULL) {
		printf("\n cannot allocate memory - matrizA\n");
		exit(1);
	}
	if ((matrizb = (int *) calloc((GRID_SIZE), sizeof(int))) == NULL) {
		printf("\n cannot allocate memory - matrizB\n");
		exit(1);
	}
	if ((matrizc = (int *) calloc((GRID_SIZE), sizeof(int))) == NULL) {
		printf("\n cannot allocate memory - matrizC\n");
		exit(1);
	}

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldnt find any platforms");
		exit(1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err < 0) {
		perror("Couldnt find any devices");
		exit(1);
	}
	/* Create the context */
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldnt create a context");
		exit(1);
	}
	/* Read program file and place content into buffer */
	program_handle = fopen(PROGRAM_FILE, "r");
	if (program_handle == NULL) {
		perror("Couldnt find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char *) malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(context, 1,
			(const char **) &program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldnt create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
				&log_size);
		program_log = (char *) malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
				log_size + 1, program_log, NULL);
		printf("%s", program_log);
		free(program_log);
		exit(1);
	}

	/* Create kernel for the mat_vec_mult function */
	kernel = clCreateKernel(program, KERNEL_FUNC, &err);
	if (err < 0) {
		perror("Couldnt create the kernel");
		exit(1);
	}
	/* Create CL buffers to hold input and output data */
	mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
	CL_MEM_COPY_HOST_PTR, sizeof(int) * GRID_SIZE, matriza, &err);
	if (err < 0) {
		perror("Couldnt create a buffer object");
		exit(1);
	}
	vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
	CL_MEM_COPY_HOST_PTR, sizeof(int) * GRID_SIZE, matrizb, NULL);
	if (err < 0) {
		perror("Couldnt create a buffer object");
		exit(1);
	}
	res_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
	CL_MEM_COPY_HOST_PTR, sizeof(int) * GRID_SIZE, matrizc, NULL);
	if (err < 0) {
		perror("Couldnt create a buffer object");
		exit(1);
	}


	/* Create kernel arguments from the CL buffers */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
	if (err < 0) {
		perror("Couldnt set the kernel argument");
		exit(1);
	}
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);

	/* Create a CL command queue for the device*/
	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0) {
		perror("Couldnt create the command queue");
		exit(1);
	}

	/* Enqueue the command queue to the device */
	work_units_per_kernel = GRID_SIZE; /* work-units per kernel */
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,
	NULL, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldnt enqueue the kernel execution command");
		exit(1);
	}

	/* Read the result */
	err = clEnqueueReadBuffer(queue, mat_buff, CL_TRUE, 0, sizeof(int) * GRID_SIZE,
			matriza, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldnt enqueue the read buffer command");
		exit(1);
	}
	err = clEnqueueReadBuffer(queue, vec_buff, CL_TRUE, 0, sizeof(int) * GRID_SIZE,
			matrizb, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldnt enqueue the read buffer command");
		exit(1);
	}
	err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(int) * GRID_SIZE,
			matrizc, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldnt enqueue the read buffer command");
		exit(1);
	}

	printf("\nMatrizA: \n");
	imprime_matriz(matriza, 32, 32);

	printf("\nMatrizB: \n");
	imprime_matriz(matrizb, 32, 32);

	printf("\nMatrizC: \n");
	imprime_matriz(matrizc, 32, 32);


	/* Deallocate resources */
	clReleaseMemObject(mat_buff);
	clReleaseMemObject(vec_buff);
	clReleaseMemObject(res_buff);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}
