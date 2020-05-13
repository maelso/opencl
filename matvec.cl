__kernel void matvec_mult(__global int* matrizA,
                          __global int* matrizB,
                          __global int* result) {

	int gid = get_global_id(0);
	unsigned long int seed = (unsigned long int)( (gid + 1111) * (gid + 1011) * (gid + 1010)) ;
	int valor; 
	
	valor = genRand(&seed, 9);
	matrizA[gid] = valor;
	
	valor = genRand(&seed, 9);
	matrizB[gid] = valor;

	result[gid] = matrizA[gid] + matrizB[gid];
}

int genRand(unsigned long int* i, int limite){
	(*i) ^= (*i) >> 12; // a
	(*i) ^= (*i) << 25; // b
	(*i) ^= (*i) >> 27; // c 
	
	int ret = ( (*i) * 2685821657736338717) % limite;
	if (ret < 0) {
   		ret *= -1;
	}
    return ret;
}
