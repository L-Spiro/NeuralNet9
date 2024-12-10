// NeuralNet9.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Utilities/NN9Utilities.h"
#include <iostream>

#include "Tensor/NN9Tensor.h"

int wmain( int _iArgC, wchar_t const * _wcpArgV[] ) {
    --_iArgC;
    std::cout << "Hello World!\n";

	auto aCode = nn9::Utilities::DownloadMnist( u"C:\\MNIST\\DownLoadTest" );

	//{
		nn9::Tensor tTensorTest( { 60, 28, 28 }, nn9::NN9_T_FLOAT, 33.2f );

		auto vView = tTensorTest.FullView<float>();
		auto vRange = tTensorTest.RangeView<float>( 28, 2 );
		{
			auto v3d = tTensorTest.Full3dView<float>();
			for ( size_t I = 0; I < vRange.size(); ++I ) {
				vRange[I] = 90.0f;
			}

			for ( int I = 0; I < 60; ++I ) {
				for ( int H = 0; H < 28; ++H ) {
					for ( int W = 0; W < 28; ++W ) {
						auto sFlat = tTensorTest.Flat( I, H, W );
						//vView[sFlat] = 90.0f;
						std::wcout << L"tTensorTest.Flat( " << I << L", " << H << L", " << W << L" ): " << sFlat << L". " << vView[sFlat] << L" " << v3d[I][H][W] << std::endl;
						break;
					}
				}
			}
		}
	//}
	
	return 0;
}
