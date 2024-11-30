// NeuralNet9.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Utilities/NN9Utilities.h"
#include <iostream>

int wmain( int _iArgC, wchar_t const * _wcpArgV[] ) {
    --_iArgC;
    std::cout << "Hello World!\n";

	auto aCode = nn9::Utilities::DownloadMnist( u"C:\\MNIST\\DownLoadTest" );
	
	return 0;
}
