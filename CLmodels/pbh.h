#ifndef PBH_H
#define PBH_H

/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */

#include "CosmoInterface/cosmointerface.h"

namespace TempLat
{

    struct ModelPars : public TempLat::DefaultModelPars {
        static constexpr size_t NScalars = 1;
        static constexpr size_t NPotTerms = 1;
    };

#define MODELNAME pbh
    template<class R>
    using Model = MakeModel(R, ModelPars);

    class MODELNAME : public Model<MODELNAME>
    {
 //...
private:

  double a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20; // Coefficients of the potential
  double hinit; // Coefficients of the potential

  public:

    MODELNAME(ParameterParser& parser, RunParameters<double>& runPar, std::shared_ptr<MemoryToolBox> toolBox): //Constructor of our model.
    Model<MODELNAME>(parser,runPar.getLatParams(), toolBox, runPar.dt, STRINGIFY(MODELLABEL)) //MODELLABEL is defined in the cmake.
    {

      /////////
      // Independent parameters of the model (read from parameters file)
      /////////
      a0 = parser.get<double>("a0");
      a1 = parser.get<double>("a1");
      a2 = parser.get<double>("a2");
      a3 = parser.get<double>("a3");
      a4 = parser.get<double>("a4");
      a5 = parser.get<double>("a5");
      a6 = parser.get<double>("a6");
      a7 = parser.get<double>("a7");
      a8 = parser.get<double>("a8");
      a9 = parser.get<double>("a9");
      a10 = parser.get<double>("a10");
      a11 = parser.get<double>("a11");
      a12 = parser.get<double>("a12");
      a13 = parser.get<double>("a13");
      a14 = parser.get<double>("a14");
      a15 = parser.get<double>("a15");
      a16 = parser.get<double>("a16");
      a17 = parser.get<double>("a17");
      a18 = parser.get<double>("a18");
      a19 = parser.get<double>("a19");
      a20 = parser.get<double>("a20");
      
      

      fldS0 = parser.get<double, 1>("initial_amplitudes");
      piS0 = parser.get<double, 1>("initial_momenta");
      
      

      alpha = 0;
      // Set as Mp
      fStar = 2.435e18;
      omegaStar = 2.435e18;

      hinit = parser.get<double>("hinit");

      setInitialPotentialAndMassesFromPotential();
        
    }

   /////////
   // Program potential (add as many functions as terms are in the potential)
   /////////

    auto potentialTerms(Tag<0>) {
      auto h = fldS(0_c); 
      auto dh = fldS(0_c) - hinit; 
      // V(dh) = a8*h^8 + a7*h^7 + ...
      return a20*pow<20>(dh) + a19*pow<19>(dh) + a18*pow<18>(dh) + a17*pow<17>(dh) + a16*pow<16>(dh) + 
             a15*pow<15>(dh) + a14*pow<14>(dh) + a13*pow<13>(dh) + a12*pow<12>(dh) + a11*pow<11>(dh) + 
             a10*pow<10>(dh) + a9*pow<9>(dh) + a8 * pow<8>(dh) + a7 * pow<7>(dh) + 
             a6 * pow<6>(dh) + a5 * pow<5>(dh) + a4 * pow<4>(dh) + a3 * pow<3>(dh) + 
             a2 * pow<2>(dh) + a1 * dh + a0;
    }

    auto potDeriv(Tag<0>) {
        auto h = fldS(0_c); // Rescale field by fStar to make it dimensionless
        auto dh = fldS(0_c) - hinit; 
        // dV/dh = 8*a8*h^7 + 7*a7*h^6 + ...
        return 20.0*a20*pow<19>(dh) + 19.0*a19*pow<18>(dh) + 18.0*a18*pow<17>(dh) + 17.0*a17*pow<16>(dh) + 16.0*a16*pow<15>(dh) + 
               15.0*a15*pow<14>(dh) + 14.0*a14*pow<13>(dh) + 13.0*a13*pow<12>(dh) + 12.0*a12*pow<11>(dh) + 11.0*a11*pow<10>(dh) + 
               10.0*a10*pow<9>(dh) + 9.0*a9*pow<8>(dh) + 8.0*a8 * pow<7>(dh) + 7.0*a7 * pow<6>(dh) + 
               6.0*a6 * pow<5>(dh) + 5.0*a5 * pow<4>(dh) + 4.0*a4 * pow<3>(dh) + 3.0*a3 * pow<2>(dh) + 
               2.0*a2 * dh + a1;
    }	

    auto potDeriv2(Tag<0>) // Second derivative with respect daughter field
    {
        auto h = fldS(0_c); // Rescale field by fStar to make it dimensionless
        auto dh = fldS(0_c) - hinit; 
        // d2V/dh2 = 56*a8*h^6 + 42*a7*h^5 + ...
        return 380.0*a20*pow<18>(dh) + 342.0*a19*pow<17>(dh) + 306.0*a18*pow<16>(dh) + 272.0*a17*pow<15>(dh) + 240.0*a16*pow<14>(dh) + 
               210.0*a15*pow<13>(dh) + 182.0*a14*pow<12>(dh) + 156.0*a13*pow<11>(dh) + 132.0*a12*pow<10>(dh) + 110.0*a11*pow<9>(dh) + 
               90.0*a10*pow<8>(dh) + 72.0*a9*pow<7>(dh) + 56.0*a8 * pow<6>(dh) + 42.0*a7 * pow<5>(dh) + 
               30.0*a6 * pow<4>(dh) + 20.0*a5 * pow<3>(dh) + 12.0*a4 * pow<2>(dh) + 6.0*a3 * dh + 
               2.0*a2;
    }
		
    };
}

#endif //LPHI4_H
