import numpy as np

def calc_del(alpha, beta):
    return 1-alpha-beta

def calc_pi_ab(alpha, beta):
    return beta/(alpha+beta), alpha/(alpha+beta)

def calc_sigma(pi_a, pi_b, mu_a, mu_b, sigma_a, sigma_b):
    return (pi_a*(sigma_a**2) + pi_b*(sigma_b**2) + pi_a*pi_b*(mu_a-mu_b)**2)**0.5

def calc_autocorr_const(pi_a, pi_b, mu_a, mu_b, sigma):
    print("pi_a = ", pi_a)
    print("pi_b = ", pi_b)
    print("sigma = ", sigma)
    return pi_a*pi_b*(mu_a-mu_b)**2 / sigma**2

def calc_nu(phi, c):
    print("c = ", c)
    print("phi = ", phi)
    d = (1+(1-2*c)*(phi**2)) / (2*phi*(1-c))
    print('d = ', d)
    return d - (d**2 - 1)**0.5

def calc_phi_vector(p, phi, nu):
    vec = []
    for i in range(0, p+1):
        vec.append((phi-nu)*(nu**(i-1)))
    return vec

def generate_ar_weights(alpha, beta, mu_a, mu_b, sigma_a, sigma_b, p):
    delta = calc_del(alpha, beta)
    pi_a, pi_b = calc_pi_ab(alpha, beta)
    sigma = calc_sigma(pi_a, pi_b, mu_a, mu_b, sigma_a, sigma_b)
    c = calc_autocorr_const(pi_a, pi_b, mu_a, mu_b, sigma)
    phi = delta
    nu = calc_nu(phi, c)

    phi_vector = calc_phi_vector(p, phi, nu)
    return phi_vector

def indicator_signal(phi_vector, return_vector, p, a=1):
    signal = 0
    #print(return_vector)
    #print(phi_vector)
    for i in range(0, p):
        phi_i =  phi_vector[i+1]
        r_t_i = return_vector[p-i-1]
        signal += a * phi_i * r_t_i
        print(phi_i, r_t_i, signal, end=" ")
        print()
    #print(np.dot(phi_vector[1:], return_vector.T))
    return signal