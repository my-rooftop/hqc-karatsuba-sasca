import time
from bp_tools import *
from data import *


def ComputerBpAw(fgtype, n, filepath, Numberoftripple, data):
    loopnum = 10
    probekeyfile = filepath + "/"
    resultfile = probekeyfile + "result/"
    
    # Output the probability distribution file
    outname_a0 = resultfile + fgtype + '_probability_y0.txt'
    outname_a1 = resultfile + fgtype + '_probability_y2.txt'
    outname_ta = resultfile + fgtype + '_probability_y1.txt'

    # Input the prior probability distribution file
    path1_probability = probekeyfile + 'mask_a0.txt'
    path2_probability = probekeyfile + 'mask_a1.txt'
    path3_probability = probekeyfile + 'mask_ta.txt'
    path4_probability = probekeyfile + 'r_a0.txt'
    path5_probability = probekeyfile + 'r_a1.txt'
    path6_probability = probekeyfile + 'r_ta.txt'

    probability1 = np.loadtxt(path1_probability)
    probability2 = np.loadtxt(path2_probability)
    probability3 = np.loadtxt(path3_probability)
    probability4 = np.loadtxt(path4_probability)
    probability5 = np.loadtxt(path5_probability)
    probability6 = np.loadtxt(path6_probability)

    in_probability1 = probability1.reshape((n, 1, Numberoftripple[0]))
    in_probability2 = probability2.reshape((n, 1, Numberoftripple[0]))
    in_probability3 = probability3.reshape((n, 1, Numberoftripple[0]))
    in_probability4 = probability4.reshape((n, 9, Numberoftripple[1]))
    in_probability5 = probability5.reshape((n, 9, Numberoftripple[1]))
    in_probability6 = probability6.reshape((n, 9, Numberoftripple[1]))

    with open(outname_a0, 'w') as f:
        f.truncate(0)

    with open(outname_a1, 'w') as f:
        f.truncate(0)

    with open(outname_ta, 'w') as f:
        f.truncate(0)

    for ff in range(n):
        print("Calculating probability distribution for Curve Group", ff, "...")
        
        # Construct the factor graph
        mrf = string2factor_graph('f1(a0,a1,ta)f2(a0,r0)f3(a0,r1)f4(a0,r2)f5(a0,r3)f6(a0,r4)f7(a0,r5)f8(a0,r6)'
                                  'f9(a0,r7)f10(a0,r8)f11(a1,r9)f12(a1,r10)f13(a1,r11)f14(a1,r12)f15(a1,r13)'
                                  'f16(a1,r14)f17(a1,r15)f18(a1,r16)f19(a1,r17)f20(ta,r18)f21(ta,r19)f22(ta,r20)'
                                  'f23(ta,r21)f24(ta,r22)f25(ta,r23)f26(ta,r24)f27(ta,r25)f28(ta,r26)'
                                  'f29(a0)f30(a1)f31(ta)f32(r0)f33(r1)f34(r2)f35(r3)f36(r4)f37(r5)f38(r6)f39(r7)'
                                  'f40(r8)f41(r9)f42(r10)f43(r11)f44(r12)f45(r13)f46(r14)f47(r15)f48(r16)f49(r17)'
                                  'f50(r18)f51(r19)f52(r20)f53(r21)f54(r22)f55(r23)f56(r24)f57(r25)f58(r26)')

        # Pass the relational constraints to the factor nodes
        f1 = factor(['a0', 'a1', 'ta'], data[1])

        f2 = factor(['a0', 'r0'], data[0])
        f3 = factor(['a0', 'r1'], data[0])
        f4 = factor(['a0', 'r2'], data[0])
        f5 = factor(['a0', 'r3'], data[0])
        f6 = factor(['a0', 'r4'], data[0])
        f7 = factor(['a0', 'r5'], data[0])
        f8 = factor(['a0', 'r6'], data[0])
        f9 = factor(['a0', 'r7'], data[0])
        f10 = factor(['a0', 'r8'], data[0])

        f11 = factor(['a1', 'r9'], data[0])
        f12 = factor(['a1', 'r10'], data[0])
        f13 = factor(['a1', 'r11'], data[0])
        f14 = factor(['a1', 'r12'], data[0])
        f15 = factor(['a1', 'r13'], data[0])
        f16 = factor(['a1', 'r14'], data[0])
        f17 = factor(['a1', 'r15'], data[0])
        f18 = factor(['a1', 'r16'], data[0])
        f19 = factor(['a1', 'r17'], data[0])

        f20 = factor(['ta', 'r18'], data[0])
        f21 = factor(['ta', 'r19'], data[0])
        f22 = factor(['ta', 'r20'], data[0])
        f23 = factor(['ta', 'r21'], data[0])
        f24 = factor(['ta', 'r22'], data[0])
        f25 = factor(['ta', 'r23'], data[0])
        f26 = factor(['ta', 'r24'], data[0])
        f27 = factor(['ta', 'r25'], data[0])
        f28 = factor(['ta', 'r26'], data[0])

        # Pass the prior probability distributions to the factor nodes
        f29 = factor(['a0'], in_probability1[ff, 0])
        f30 = factor(['a1'], in_probability2[ff, 0])
        f31 = factor(['ta'], in_probability3[ff, 0])

        f32 = factor(['r0'], in_probability4[ff, 0])
        f33 = factor(['r1'], in_probability4[ff, 1])
        f34 = factor(['r2'], in_probability4[ff, 2])
        f35 = factor(['r3'], in_probability4[ff, 3])
        f36 = factor(['r4'], in_probability4[ff, 4])
        f37 = factor(['r5'], in_probability4[ff, 5])
        f38 = factor(['r6'], in_probability4[ff, 6])
        f39 = factor(['r7'], in_probability4[ff, 7])
        f40 = factor(['r8'], in_probability4[ff, 8])

        f41 = factor(['r9'], in_probability5[ff, 0])
        f42 = factor(['r10'], in_probability5[ff, 1])
        f43 = factor(['r11'], in_probability5[ff, 2])
        f44 = factor(['r12'], in_probability5[ff, 3])
        f45 = factor(['r13'], in_probability5[ff, 4])
        f46 = factor(['r14'], in_probability5[ff, 5])
        f47 = factor(['r15'], in_probability5[ff, 6])
        f48 = factor(['r16'], in_probability5[ff, 7])
        f49 = factor(['r17'], in_probability5[ff, 8])

        f50 = factor(['r18'], in_probability6[ff, 0])
        f51 = factor(['r19'], in_probability6[ff, 1])
        f52 = factor(['r20'], in_probability6[ff, 2])
        f53 = factor(['r21'], in_probability6[ff, 3])
        f54 = factor(['r22'], in_probability6[ff, 4])
        f55 = factor(['r23'], in_probability6[ff, 5])
        f56 = factor(['r24'], in_probability6[ff, 6])
        f57 = factor(['r25'], in_probability6[ff, 7])
        f58 = factor(['r26'], in_probability6[ff, 8])

        mrf.change_factor_distribution('f1', f1)

        mrf.change_factor_distribution('f2', f2)
        mrf.change_factor_distribution('f3', f3)
        mrf.change_factor_distribution('f4', f4)
        mrf.change_factor_distribution('f5', f5)
        mrf.change_factor_distribution('f6', f6)
        mrf.change_factor_distribution('f7', f7)
        mrf.change_factor_distribution('f8', f8)
        mrf.change_factor_distribution('f9', f9)
        mrf.change_factor_distribution('f10', f10)

        mrf.change_factor_distribution('f11', f11)
        mrf.change_factor_distribution('f12', f12)
        mrf.change_factor_distribution('f13', f13)
        mrf.change_factor_distribution('f14', f14)
        mrf.change_factor_distribution('f15', f15)
        mrf.change_factor_distribution('f16', f16)
        mrf.change_factor_distribution('f17', f17)
        mrf.change_factor_distribution('f18', f18)
        mrf.change_factor_distribution('f19', f19)

        mrf.change_factor_distribution('f20', f20)
        mrf.change_factor_distribution('f21', f21)
        mrf.change_factor_distribution('f22', f22)
        mrf.change_factor_distribution('f23', f23)
        mrf.change_factor_distribution('f24', f24)
        mrf.change_factor_distribution('f25', f25)
        mrf.change_factor_distribution('f26', f26)
        mrf.change_factor_distribution('f27', f27)
        mrf.change_factor_distribution('f28', f28)

        mrf.change_factor_distribution('f29', f29)
        mrf.change_factor_distribution('f30', f30)
        mrf.change_factor_distribution('f31', f31)

        mrf.change_factor_distribution('f32', f32)
        mrf.change_factor_distribution('f33', f33)
        mrf.change_factor_distribution('f34', f34)
        mrf.change_factor_distribution('f35', f35)
        mrf.change_factor_distribution('f36', f36)
        mrf.change_factor_distribution('f37', f37)
        mrf.change_factor_distribution('f38', f38)
        mrf.change_factor_distribution('f39', f39)
        mrf.change_factor_distribution('f40', f40)

        mrf.change_factor_distribution('f41', f41)
        mrf.change_factor_distribution('f42', f42)
        mrf.change_factor_distribution('f43', f43)
        mrf.change_factor_distribution('f44', f44)
        mrf.change_factor_distribution('f45', f45)
        mrf.change_factor_distribution('f46', f46)
        mrf.change_factor_distribution('f47', f47)
        mrf.change_factor_distribution('f48', f48)
        mrf.change_factor_distribution('f49', f49)

        mrf.change_factor_distribution('f50', f50)
        mrf.change_factor_distribution('f51', f51)
        mrf.change_factor_distribution('f52', f52)
        mrf.change_factor_distribution('f53', f53)
        mrf.change_factor_distribution('f54', f54)
        mrf.change_factor_distribution('f55', f55)
        mrf.change_factor_distribution('f56', f56)
        mrf.change_factor_distribution('f57', f57)
        mrf.change_factor_distribution('f58', f58)

        # Use LLR-NMS
        llr_bp = llr_belief_propagation(mrf, True, 0.8)

        posterior_a0 = llr_bp.belief('a0', loopnum)
        posterior_a1 = llr_bp.belief('a1', loopnum)
        posterior_ta = llr_bp.belief('ta', loopnum)

        count_a = posterior_a0.get_distribution().reshape(-1, 2)
        count_b = posterior_a1.get_distribution().reshape(-1, 2)
        count_c = posterior_ta.get_distribution().reshape(-1, 2)

        with open(outname_a0, 'a') as f_a:
            np.savetxt(f_a, count_a)

        with open(outname_a1, 'a') as f_b:
            np.savetxt(f_b, count_b)

        with open(outname_ta, 'a') as f_c:
            np.savetxt(f_c, count_c)


if __name__ == '__main__':
    nt = [2, 2]
    Numberoftripple = np.array(nt)
    filepath = "./predictions"
    n = 9216

    start_time = time.time()
    ComputerBpAw("HQC", n, filepath, Numberoftripple, data)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.2f} seconds")

    print("\nBelief Propagation calculation is completed")
