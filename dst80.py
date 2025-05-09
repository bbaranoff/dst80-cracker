#!/usr/bin/env python3
"""
OpenCL-accelerated DST-80 key search

Usage:
  python dst80.py <challenge_hex> <target_signature_hex> <max_keys>

Dependencies:
  pip install numpy pyopencl
"""
import argparse
import numpy as np
import pyopencl as cl
import dst80_verif  # Python reference for verification
from multiprocessing import Process, Manager

# ---------------------------
# Configuration
# ---------------------------
CHUNK = 1024    # number of keys per batch
NUM_PROCS = 32  # number of parallel worker processes

# ---------------------------
# OpenCL initialization
# ---------------------------
def init_opencl():
    # Load and build the OpenCL kernel
    with open('dst80_kernel.cl', 'r') as f:
        src = f.read()
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, src).build()
    # Allocate device buffers for up to CHUNK elements
    buf_l = cl.Buffer(ctx, cl.mem_flags.READ_ONLY,  size=8 * CHUNK)
    buf_r = cl.Buffer(ctx, cl.mem_flags.READ_ONLY,  size=8 * CHUNK)
    buf_c = cl.Buffer(ctx, cl.mem_flags.READ_ONLY,  size=8 * CHUNK)
    buf_o = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=4 * CHUNK)
    return queue, program, buf_l, buf_r, buf_c, buf_o

# ---------------------------
# Key packing
# ---------------------------
def make_kl(i, j, k):
    # pack 5-byte keyl: bytes [i,j,k,0xAA,0xAA]
    return ((i.astype(np.uint64) << 32) |
            (j.astype(np.uint64) << 24) |
            (k.astype(np.uint64) << 16) |
            (np.uint64(0xAA)         <<  8) |
             np.uint64(0xAA))

def make_kr(i, j, k):
    # pack 5-byte keyr: bytes [0xAA,0xAA,255-k,255-j,255-i]
    return ((np.uint64(0xAA) << 32) |
            (np.uint64(0xAA) << 24) |
            ((255 - k).astype(np.uint64) << 16) |
            ((255 - j).astype(np.uint64) <<  8) |
             (255 - i).astype(np.uint64))

# ---------------------------
# Batch compute via OpenCL
# ---------------------------
def compute_chunk(queue, program, buf_l, buf_r, buf_c, buf_o, kl_arr, kr_arr, C):
    n = kl_arr.size
    # prepare challenge array
    chal = np.full(n, np.uint64(C), dtype=np.uint64)
    # copy to device
    cl.enqueue_copy(queue, buf_l, kl_arr)
    cl.enqueue_copy(queue, buf_r, kr_arr)
    cl.enqueue_copy(queue, buf_c, chal)
    # launch kernel
    program.dst80_kernel(queue, (n,), None, buf_l, buf_r, buf_c, buf_o)
    # read back results
    out = np.empty(n, dtype=np.uint32)
    cl.enqueue_copy(queue, out, buf_o)
    queue.finish()
    return out

# ---------------------------
# Worker process
# ---------------------------
def worker(pid, challenge, target, max_keys, stop_event, result_dict):
    """Worker process: prints debug info on the first iteration and searches chunks"""
    # initialize OpenCL locally (per process)
    queue, program, buf_l, buf_r, buf_c, buf_o = init_opencl()
    stride = NUM_PROCS * CHUNK
    start = pid * CHUNK
    # Debug: announce worker and first base
    #print(f"Worker {pid} started: processing keys starting at index {start}")

    # iterate over chunks assigned to this pid
    for base in range(start, max_keys, stride):
        if stop_event.is_set():
            return
        sz = min(CHUNK, max_keys - base)
        # Debug: show this chunk
        #print(f"Worker {pid}: chunk base={base}, size={sz}")
        idx = np.arange(base, base + sz, dtype=np.uint32)
        i = idx % 255
        j = (idx // 255) % 255
        k = (idx // (255 * 255)) % 255

        # pack keys
        kl = make_kl(i, j, k)
        kr = make_kr(i, j, k)

                # compute signatures
        sigs = compute_chunk(queue, program, buf_l, buf_r, buf_c, buf_o,
                             kl, kr, challenge)
        # Debug: on first iteration, print the returned sigs array
        if base == start:
            #print(f"Worker {pid} first-iteration sigs: {sigs.tolist()}")
            # Check first signature for pid 0 against target
            if pid == 0:
                first_sig = int(sigs[0])
                keyl0 = int(kl[0])
                keyr0 = int(kr[0])
                print(f"PID 0 first sig=0x{first_sig:06x}, keyl=0x{keyl0:010x}, keyr=0x{keyr0:010x}")
                if first_sig == target:
                    print("First iteration matches target!")
                else:
                    print("First iteration DOES NOT match target.")

        # check for matches
        for loc, sig in enumerate(sigs.tolist()):
            if sig != target:
                continue
            # verify against Python reference implementation
            keyl_int = int(kl[loc])
            keyr_int = int(kr[loc])
            if dst80_verif.dst80(keyl_int, keyr_int, challenge) != target:
                continue
            # Print match info including pid
            print(f"Found verified match @pid={pid}: keyl=0x{keyl_int:010x}, keyr=0x{keyr_int:010x}, sig=0x{sig:06x}, target=0x{target:06x}")
            result_dict['kl'] = f"{keyl_int:010x}"
            result_dict['kr'] = f"{keyr_int:010x}"
            stop_event.set()
            return
    # initialize OpenCL locally (per process)
    queue, program, buf_l, buf_r, buf_c, buf_o = init_opencl()
    stride = NUM_PROCS * CHUNK
    start = pid * CHUNK

    # iterate over chunks assigned to this pid
    for base in range(start, max_keys, stride):
        if stop_event.is_set():
            return
        sz = min(CHUNK, max_keys - base)
        idx = np.arange(base, base + sz, dtype=np.uint32)
        i = idx % 255
        j = (idx // 255) % 255
        k = (idx // (255 * 255)) % 255

        # pack keys
        kl = make_kl(i, j, k)
        kr = make_kr(i, j, k)

        # compute signatures
        sigs = compute_chunk(queue, program, buf_l, buf_r, buf_c, buf_o,
                             kl, kr, challenge)

                # check for matches
        for loc, sig in enumerate(sigs.tolist()):
            if sig != target:
                continue
            # verify against Python reference implementation
            keyl_int = int(kl[loc])
            keyr_int = int(kr[loc])
            if dst80_verif.dst80(keyl_int, keyr_int, challenge) != target:
                continue
            # Print match info including pid
            #print(f"Found verified match @pid={pid}: keyl=0x{keyl_int:010x}, keyr=0x{keyr_int:010x}, sig=0x{sig:06x}, target=0x{target:06x}")
            result_dict['kl'] = f"{keyl_int:010x}"
            result_dict['kr'] = f"{keyr_int:010x}"
            stop_event.set()
            return

# ---------------------------
# Main entry point
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCL DST-80 key search')
    parser.add_argument('challenge',  help='Hex challenge, e.g. 0xC212345678')
    parser.add_argument('target',     help='Hex target sig, e.g. 0xb01946')
    parser.add_argument('max_keys', type=int, help='Max keys to search')
    args = parser.parse_args()

    C = int(args.challenge, 16)
    T = int(args.target, 16)
    M = args.max_keys

    mgr   = Manager()
    stop  = mgr.Event()
    res   = mgr.dict()

    # Spawn worker processes and print their OS PIDs
    procs = []
    for pid in range(NUM_PROCS):
        p = Process(target=worker,
                    args=(pid, C, T, M, stop, res))
        p.start()
        #print(f"Spawned worker {pid} as OS PID {p.pid}")
        procs.append(p)

    # Wait for all to finish
    for p in procs:
        p.join()

    # Report result
    if 'kl' in res:
        print(f"Match: keyl={res['kl']}, keyr={res['kr']}, sig=0x{T:06x}")
    else:
        print(f"No match after searching {M} keys.")
