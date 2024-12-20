// NeuralNet9.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Foundation/NN9Intrin.h"
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
			nn9::Math::Mul( vView, 200.0 );
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

			double iVales[64] = {
				-1, 2, DBL_MAX, 0, -FLT_MAX, 256, 7, 8, 9, 10,
				120, 221, 322, 423, 124, 125, 126, 127, 128, 129,
				0x7F, -1, 77, -0x7F, 1, 64, 254, -500, -500, 500,
				0x7F, INT16_MIN, 32, -0x7F, 1, 64, 254, -500, -500, 500,
				0x7F, -1, 32, -0x7F, 1, 64, 254, -500, -500, 500,
				0x7F, -1, 32, -0x7F, 1, 64, 254, -500, -500, 500,
				45, 66
			};
			//__m512i mVal = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(iVales) );
			//__m256i mVal = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(iVales) );
			//__m512 mVal = _mm512_loadu_ps( iVales );
			__m512d mVal = _mm512_loadu_pd( iVales );
			float ui16Dst[64];
			nn9::Intrin::double_scast( mVal, ui16Dst );

			nn9::Tensor tBFloat16 = tTensorTest.CopyAs( nn9::NN9_T_FLOAT16 );
			auto vViewBf16 = tBFloat16.FullView<nn9::float16>();
			/*nn9::Tensor tBFloat16 = tTensorTest.CopyAs( nn9::NN9_T_BFLOAT16 );
			auto vViewBf16 = tBFloat16.FullView<bfloat16_t>();*/
			/*nn9::Tensor tBFloat16 = tTensorTest.CopyAs( nn9::NN9_T_INT32 );
			auto vViewBf16 = tBFloat16.FullView<int32_t>();*/

			nn9::Tensor tBackToFloat = tBFloat16.CopyAs( nn9::NN9_T_FLOAT );
			auto vViewNewFloat = tBackToFloat.FullView<float>();
			nn9::Timer tTimer;


			double dSum = 0.0;
			tTimer.Start();
			for ( int i = 0; i < 50000; ++i ) {
				//dSum += nn9::Math::Sum( vViewBf16 );
				nn9::Math::Abs( vViewNewFloat );
			}

			tTimer.Stop();
			std::wcout << L"nn9::Math::Abs( float ): " << tTimer.ElapsedSeconds() << L". " << dSum << std::endl;
			tTimer.Reset();


			dSum = 0.0;
			
			tTimer.Start();
			for ( int i = 0; i < 50000; ++i ) {
				//dSum += nn9::Math::KahanSum( vViewNewFloat );
				nn9::Math::Abs( vViewBf16 );
			}

			tTimer.Stop();
			std::wcout << L"nn9::Math::Abs( nn9::float16 ): " << tTimer.ElapsedSeconds() << L". " << dSum << std::endl;
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
