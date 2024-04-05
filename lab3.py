﻿from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def linear_approx(x: list, y: list):
    coeffs = np.transpose(np.array([[c for c in x],[1 for _ in range(len(x))]]));
    results = np.transpose(np.array([v for v in y]));

    
    coeffs_T = np.transpose(coeffs);

    normal_coeffs = coeffs_T.dot(coeffs);
    normal_result = coeffs_T.dot(results);
    
    return np.linalg.solve(normal_coeffs, normal_result);
    
def quadratic_approx(x: list, y: list):
    m = 2;
    results = [0 for _ in range(m+1)];
    
    for i in range(m, -1, -1):
        for j in range(len(x)):
            results[m-i] += y[j] * pow(x[j], i);

    coeffs = [[0 for _ in range(m+1)] for _ in range(m+1)]
    for i in range(m, -1, -1):
        for j in range(m, -1, -1):
            for k in range(len(x)):
                coeffs[m-i][m-j] += pow(x[k], i+j);
    
    coeffs_arr = np.array(coeffs)
    results_arr = np.array(results)
    
    return np.linalg.solve(coeffs_arr, np.transpose(results_arr));
    
def mnk_deviation(x: list, y: list):
    P = 0;
    
    Sx = 0
    Sy = 0
    
    div = 0;
    for i in range(len(x)):
        Sx += x[i];
        Sy += y[i];
    
        P += x[i]**2;
        P -= y[i]**2;
    
        div += x[i]*y[i];
    
    x2 = (Sx**2)/len(x);
    y2 = (Sy**2)/len(x);
    
    P -= x2;
    P += y2;

    div -= Sx*Sy/len(x);
    
    P /= div;
    
    B1 = [(-P - sqrt(P**2 + 4))/2,
          (-P + sqrt(P**2 + 4))/2]
    
    B0 = [0, 0];

    for c in range(len(B1)):
        t = 0;
        for i in range(len(y)):
            B0[c] += y[i];
        for i in range(len(x)):
            t += x[i];
        t *= B1[c];
        B0[c] -= t;
        B0[c] /= len(x);

    sigma_K = [0 for _ in range(len(B0))];
    tmp = 0;
    div = 0;
    for i in range(len(B0)):
        for n in range(len(x)):
            tmp += B1[i] * x[n];
            tmp -= y[n];
            tmp += B0[i]**2;
            div += B1[i]**2 + 1;
            tmp /= div;
            sigma_K[i] += tmp;
            tmp = 0;
            div = 0;
        sigma_K[i] /= len(x);
        sigma_K[i] **= (1/2);
  
    return [B0, B1];
    
    

# def linear_least_squares(x, y):
#     n = len(x)
#     A = np.vstack([x, np.ones(n)]).T
#     m, c = np.linalg.lstsq(A, y, rcond=None)[0]
#     def f(x):
#         return x*m + c;
#     return f

# def quadratic_approximation(x, y):
#     n = len(x)
#     A = np.vstack([np.ones(n), x, x**2]).T
#     a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
#     def f(x):
#         return a + b*x + c*x**2;
#     return f

def plot_function(func, x_range, label, _type = '-'):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = func(x)
    
    plt.plot(x, y, _type, label=label)
    #plt.title(title)
    plt.grid(True)

def linear_func(a: float, b: float):
    def f(x):
        return a*x + b;
    return f;

def quadratic_func(a: float, b: float, c: float):
    def f(x):
        return a*x**2 + b*x + c;
    return f;
# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([1, 3, 7, 9, 11, 14])

# plot_function(linear_least_squares(x, y), (-10, 10))

# plot_function(quadratic_approximation(x, y), (-10, 10))

# print(linear_approx([1, 2, 3], [1, 2, 2]))
# print(quadratic_approx([1, 2, 3, 4, 5], [5, 15, 25, 45, 65]))
# print(mnk_deviation([-2, -1, 1, 2], [-0.8, 0.4, 1.0, -0.6]))

if __name__ == "__main__":
    n = 10;
    x = [-2, -1, 0, 1, 2];
    y = [-2*n - 1, -n+1, n+4, -n-1, 3*n+1];
    
    linear_coeffs = linear_approx(x, y);
    quadratic_coeffs = quadratic_approx(x, y);

    print("Коэффиценты: ")
    print(f"\tЛинейной аппроксимации: {linear_coeffs}")
    print(f"\tКвадратичной аппроксимации: {quadratic_coeffs}")
    plot_function(linear_func(*linear_coeffs), [x[0], x[-1]], 'Линейная аппроксимация', 'y-')
    plot_function(quadratic_func(*quadratic_coeffs), [x[0], x[-1]], 'Квадратичная аппроксимация', 'b--',)
    plt.plot(x, y, 'rx', label='Табличные точки')
    plt.title('Линейная и квадратичная аппроксимации')
    plt.legend()
    plt.grid(True)
    plt.show();
    #================================================================
    r = [8, 9, 10, 11, 30];
    t = [20.3, 21.5, 22.1, 20.5, 19.8];
    coeffs = mnk_deviation(r, t);

    print("Коэффиценты: ")
    print(f"\tB0: {coeffs[0]}")
    print(f"\tB1: {coeffs[1]}")
    plt.plot(r, t, 'rx', label='Табличные точки')
    plot_function(linear_func(coeffs[1][0], coeffs[0][0]), [r[0], r[-1]], 'Линейная аппроксимация 1', 'y-')
    plot_function(linear_func(coeffs[1][1], coeffs[0][1]), [r[0], r[-1]], 'Линейная аппроксимация 2', 'g-')
    plt.legend()
    plt.grid(True)
    plt.show();
    # plt.figure(figsize=(10, 6))
   
    # plt.plot(x, linear_func(*linear_coeffs), 'b--', label=f'Линейная аппроксимация\n$a_0={linear_coeffs[0]:.2f}, a_1={linear_coeffs[1]:.2f}')  # Линейная модель
    # plt.plot(x, quadratic_func(*quadratic_coeffs), 'g-', label=f'Квадратичная аппроксимация\n$b_0={quadratic_coeffs[0]:.2f}, b_1={quadratic_coeffs[1]:.2f}, b_2={quadratic_coeffs[2]:.2f}')  # Квадратичная модель
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Линейная и квадратичная аппроксимация методом наименьших квадратов')
    # plt.legend()
    # plt.grid(True)
    # plt.show()