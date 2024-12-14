// NeuralNet9.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Ops/NN9Init.h"
#include "Ops/NN9Math.h"
#include "Utilities/NN9Timer.h"
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
		nn9::Init::OrthogonalInitialization( 60, 28 * 28, vView );
		nn9::Math::Asinh( vView );
		auto vRange = tTensorTest.RangeView<float>( 28, 200 );

		{
			auto v3d = tTensorTest.Full3dView<float>();
			for ( size_t I = 0; I < vRange.size(); ++I ) {
				vRange[I] = 90.0f;
			}

			//for ( int I = 0; I < 60; ++I ) {
			//	for ( int H = 0; H < 28; ++H ) {
			//		for ( int W = 0; W < 28; ++W ) {
			//			auto sFlat = tTensorTest.Flat( I, H, W );
			//			//vView[sFlat] = 90.0f;
			//			std::wcout << L"tTensorTest.Flat( " << I << L", " << H << L", " << W << L" ): " << sFlat << L". " << vView[sFlat] << L" " << v3d[I][H][W] << std::endl;
			//			break;
			//		}
			//	}
			//}

			nn9::float16 f16Neg = -0.124;

			nn9::Tensor tBFloat16 = tTensorTest.CopyAs( nn9::NN9_T_FLOAT16 );
			auto vViewBf16 = tBFloat16.FullView<nn9::float16>();
			/*nn9::Tensor tBFloat16 = tTensorTest.CopyAs( nn9::NN9_T_BFLOAT16 );
			auto vViewBf16 = tBFloat16.FullView<bfloat16_t>();*/

			nn9::Tensor tBackToFloat = tBFloat16.CopyAs( nn9::NN9_T_FLOAT );
			auto vViewNewFloat = tBackToFloat.FullView<float>();
			nn9::Timer tTimer;


			double dSum = 0.0;
			tTimer.Start();
			for ( int i = 0; i < 50000; ++i ) {
				//dSum += nn9::Math::Sum( vViewBf16 );
				nn9::Math::Abs( v3d );
			}

			tTimer.Stop();
			std::wcout << L"nn9::Math::Acos( bfloat16_t ): " << tTimer.ElapsedSeconds() << L". " << dSum << std::endl;
			tTimer.Reset();


			dSum = 0.0;
			tTimer.Start();
			for ( int i = 0; i < 50000; ++i ) {
				//dSum += nn9::Math::KahanSum( vViewNewFloat );
				nn9::Math::Abs( vViewBf16 );
			}

			tTimer.Stop();
			std::wcout << L"nn9::Math::Acos( float ): " << tTimer.ElapsedSeconds() << L". " << dSum << std::endl;
			/*for ( int W = 0; W <= vViewNewFloat.size() - 16; W += 16 ) {
				std::wcout << vViewNewFloat[W+0] << L" " << vViewNewFloat[W+1] << L" " << vViewNewFloat[W+2] << L" " << vViewNewFloat[W+3] << L" " << vViewNewFloat[W+4] << L" " << vViewNewFloat[W+5] << L" " << vViewNewFloat[W+6] << L" " << vViewNewFloat[W+7] << L" "
					<< vViewNewFloat[W+8] << L" " << vViewNewFloat[W+9] << L" " << vViewNewFloat[W+10] << L" " << vViewNewFloat[W+11] << L" " << vViewNewFloat[W+12] << L" " << vViewNewFloat[W+13] << L" " << vViewNewFloat[W+14] << L" " << vViewNewFloat[W+15] << std::endl;
			}*/
			for ( int W = 0; W <= vViewBf16.size() - 16; W += 16 ) {
				std::wcout << (float)vViewBf16[W+0] << L" " << (float)vViewBf16[W+1] << L" " << (float)vViewBf16[W+2] << L" " << (float)vViewBf16[W+3] << L" " << (float)vViewBf16[W+4] << L" " << (float)vViewBf16[W+5] << L" " << (float)vViewBf16[W+6] << L" " << (float)vViewBf16[W+7] << L" "
					<< (float)vViewBf16[W+8] << L" " << (float)vViewBf16[W+9] << L" " << (float)vViewBf16[W+10] << L" " << (float)vViewBf16[W+11] << L" " << (float)vViewBf16[W+12] << L" " << (float)vViewBf16[W+13] << L" " << (float)vViewBf16[W+14] << L" " << (float)vViewBf16[W+15] << std::endl;
			}

		}
	//}
	
	return 0;
}
