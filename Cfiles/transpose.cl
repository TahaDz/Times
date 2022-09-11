// Data size = 20
__kernel void transpose( const __global float* input, 
			__global float* output,  
			__global float *buffer, 
			const int P, const int Q) {
			
    const int TRANSPOSEX = P; const int TRANSPOSEY = Q;                

    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q
 
    // Set-up the local memory for shuffling
    //__local float buffer[TRANSPOSEX][TRANSPOSEY];// Q, P
 
    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty* TRANSPOSEX + tx] = input[ID1*P + ID0];
    }
 
    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;
 
    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx*TRANSPOSEX+ty];
    }
}

