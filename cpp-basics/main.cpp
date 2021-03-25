#include <cmath>    // STD Math functions
#include <vector>   // Vector containers
#include <iostream> // Input/output

int main(void)
{
    std::cout << "Hello World!" << std::endl;

    int n = 10;
    double x_min = 0.0;
    double x_max = 1.0;

    std::vector<double> x(n, 0.0); // Using the vector constructor
    std::vector<double> f_x;       // Unasigned

    f_x.resize(n); // Setting the size of the vector

    for (int i = 0; i < n; i++)
    {
        x[i] = x_min + (x_max - x_min) * i / (n - 1.0);
        f_x[i] = std::cos(x[i]);
    }

    for (int i = 0; i < n; i++)
    {
        std::cout << "x[" << i << "] = " << x[i] << "; "
                  << "f_x[" << i << "] = " << f_x[i] << "; " << std::endl;
    }

    return 0;
}