#include <string.h>
#include <stdio.h>
#include "fitsio.h"

#include <iostream>
#include <sstream>

#include <bitset>
#include <sys/time.h>
#include <time.h>
#include <inttypes.h>
#include <fstream>
#include <unistd.h>
#include <getopt.h>    /* for getopt_long; standard getopt is in unistd.h */
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <ctime>
#include <climits>
#include <cmath>
#include <iomanip>
#include <sys/resource.h>

#include "globalConstants.h"


#include "TFile.h"
#include "TNtuple.h"
#include "TObject.h"
#include "tinyxml2.h"
#include "gConfig.h"
#include "gCal.h"
#include <unistd.h>

using namespace std;


struct ccdGeom_t
{
    int nrow;
    int ncol;
    int npre;
    int binrow;
    int bincol;
    ccdGeom_t(int nrow=1e7, int ncol=1e7, int npre=7, int binrow=1, int bincol=1):nrow(nrow), ncol(ncol), npre(npre), binrow(binrow), bincol(bincol) {};
};


int deleteFile(const char *fileName){
    cout << yellow;
    cout << "Will overwrite: " << fileName << endl << endl;
    cout << normal;
    return unlink(fileName);
}

bool fileExist(const char *fileName){
    ifstream in(fileName,ios::in);

    if(in.fail()){
        //cout <<"\nError reading file: " << fileName <<"\nThe file doesn't exist!\n\n";
        in.close();
        return false;
    }

    in.close();
    return true;
}

/*========================================================
  ASCII progress bar
  ==========================================================*/
void showProgress(unsigned int currEvent, unsigned int nEvent) {

    const int nProgWidth=50;

    if ( currEvent != 0 ) {
        for ( int i=0;i<nProgWidth+8;i++)
            cout << "\b";
    }

    double percent = (double) currEvent/ (double) nEvent;
    int nBars = (int) ( percent*nProgWidth );

    cout << " |";
    for ( int i=0;i<nBars-1;i++)
        cout << "=";
    if ( nBars>0 )
        cout << ">";
    for ( int i=nBars;i<nProgWidth;i++)
        cout << " ";
    cout << "| " << setw(3) << (int) (percent*100.) << "%";
    cout << flush;

}

void printCopyHelp(const char *exeName, bool printFullHelp=false){

    if(printFullHelp){
        cout << bold;
        cout << endl;
        cout << "This program extracts hits and tracks, computes their relevant parameters\n";
        cout << "and saves them in a root file.\n";
        cout << normal;
    }
    cout << "==========================================================================\n";
    cout << yellow;
    cout << "\nUsage:\n";
    cout << "  "   << exeName << " <input file> \n\n";
    cout << "\nOptions:\n";
    cout << "  -o <optional output filename> to specify the output file name.\n";
    cout << "  -c <extract config xml file> to specify the config file to use.\n";
    cout << "     If missing, \"extractConfig.xml\" from the current dir will be used.\n";
    cout << "  -C <extract calibration xml file> to specify the calibration file to use, if separate.\n";
    cout << "  -q for quiet (no screen output)\n";
    cout << "  -m <optional mask file> to provide a bad pixels mask\n";
    cout << "  -p <optional mask file> to provide a partial bad pixels mask, which will be or-ed with the internally computed one\n";
    cout << "  -a internally compute the bad pixels mask\n";
    cout << "  -s <HDU number> for processing a single HDU \n";
    cout << "  -b <integer> number of pixels for halo mask - overrides XML config \n";
    cout << "  -t <integer> pixel threshold for halo mask \n";
    cout << "  -H, -h this help message \n\n";
    cout << normal;
    cout << blue;
    cout << "For any problems or bugs contact Javier Tiffenberg <javiert@fnal.gov>\n\n";
    cout << normal;
    cout << "==========================================================================\n\n";
}

int fitsHeaderToTree(fitsfile *fptr, TFile *outRootFile){
    int status = 0;   /*  CFITSIO status value MUST be initialized to zero!  */
    int nhdu   = 0;
    outRootFile->cd();
    int hdutype;
    if(status!=0) return -2;
    const int maxStringSize = 256;
    char keyName[maxStringSize];
    char keyValue[maxStringSize];
    char comment[maxStringSize];

    fits_get_num_hdus(fptr, &nhdu, &status); // get the number of HDUs
    for(int eI=1; eI<=nhdu; ++eI){  /* Main loop through each extension */
        fits_movabs_hdu(fptr, eI, &hdutype, &status);
        int nKeys = 0;
        fits_get_hdrspace(fptr, &nKeys, 0, &status);
        std::vector<string> vKeyName;
        std::vector<string> vKeyValue;
        std::vector<string> vComment;
        for (int i = 0; i < nKeys; ++i){
			// fits_read_keyn 1-indexes the keywords, so start at 1 (0 resets to start)
            fits_read_keyn(fptr, i+1, keyName, keyValue, comment, &status);
            vKeyName.push_back(keyName);
            vKeyValue.push_back(keyValue);
            if(vKeyValue.back()[0] == '\'') vKeyValue.back().erase(0,1);
            if(vKeyValue.back()[vKeyValue.back().size()-1] == '\'') vKeyValue.back().erase(vKeyValue.back().size()-1,1);
            vComment.push_back(comment);
        }
        ostringstream headerTreeName;
        headerTreeName << "headerTree_" << eI-1; 
        TTree headerTree(headerTreeName.str().c_str(),headerTreeName.str().c_str());
        for (int i = 0; i < nKeys; ++i){
            headerTree.Branch((vKeyName[i]).c_str(),(void*)(vKeyValue[i].c_str()),"string/C",maxStringSize);
        }
        headerTree.Fill();
        headerTree.Write();  
    }
    return 0;
}

string bitpix2TypeName(int bitpix){

    string typeName;
    switch(bitpix) {
        case BYTE_IMG:
            typeName = "BYTE(8 bits)";
            break;
        case SHORT_IMG:
            typeName = "SHORT(16 bits)";
            break;
        case LONG_IMG:
            typeName = "INT(32 bits)";
            break;
        case FLOAT_IMG:
            typeName = "FLOAT(32 bits)";
            break;
        case DOUBLE_IMG:
            typeName = "DOUBLE(64 bits)";
            break;
        default:
            typeName = "UNKNOWN";
    }
    return typeName;
}


struct track_t{
    //   TNtuple &nt;
    vector<Int_t>    xPix;
    vector<Int_t>    yPix;
    vector<Float_t>  ePix; //actually not ADC counts - this has units of e-
    vector<Int_t>    ePixInt; //ePix rounded

    Int_t flag;
    Int_t nSat;
    Int_t id;

    Int_t xMin;
    Int_t xMax;
    Int_t yMin;
    Int_t yMax;
    Float_t distance;
    Float_t bleedX;
    Float_t bleedY;
    Float_t clusterDist;

    track_t() : flag(0), nSat(0), xMin(1e7), xMax(0), yMin(1e7), yMax(0), distance(1e7), bleedX(1e7), bleedY(1e7), clusterDist(1e7) {};
    void fill(const Int_t &x , const Int_t &y, const Float_t &ePixVal, const Int_t &ePixIntVal){ xPix.push_back(x); yPix.push_back(y); ePix.push_back(ePixVal); ePixInt.push_back(ePixIntVal); };

    void reset(){ xPix.clear(); yPix.clear(); ePix.clear(); ePixInt.clear();
        flag=0; nSat=0; xMin=1e7; xMax=0; yMin=1e7; yMax=0; distance=1e7; bleedX=1e7; bleedY=1e7; clusterDist=1e7; };
};

struct hitTreeEntry_t{
    track_t hit;
    //Int_t   runID;
    Int_t   ohdu;
    //Int_t   expoStart;
    Int_t   nSavedPix;

    Float_t *hitParam;

    hitTreeEntry_t(const Int_t nPar): ohdu(-1), nSavedPix(0){ hitParam = new Float_t[nPar]; };
    ~hitTreeEntry_t(){ delete[] hitParam; };
};

void copyTrack(const hitTreeEntry_t &src, hitTreeEntry_t &dest) {
    //dest.runID = src.runID;
    //dest.expoStart = src.expoStart;
    dest.ohdu = src.ohdu;
    dest.nSavedPix = src.nSavedPix;
    for (int i=0; i<gNExtraTNtupleVars; i++) {
        dest.hitParam[i] = src.hitParam[i];
    }
    dest.hit.flag = src.hit.flag;
    dest.hit.nSat = src.hit.nSat;
    dest.hit.id = src.hit.id;
    dest.hit.xMin = src.hit.xMin;
    dest.hit.xMax = src.hit.xMax;
    dest.hit.yMin = src.hit.yMin;
    dest.hit.yMax = src.hit.yMax;
    dest.hit.distance = src.hit.distance;
    dest.hit.bleedX = src.hit.bleedX;
    dest.hit.bleedY = src.hit.bleedY;
    dest.hit.clusterDist = src.hit.clusterDist;

    dest.hit.xPix.clear();
    dest.hit.yPix.clear();
    dest.hit.ePix.clear();

    std::copy(src.hit.xPix.begin(), src.hit.xPix.end(), back_inserter(dest.hit.xPix));
    std::copy(src.hit.yPix.begin(), src.hit.yPix.end(), back_inserter(dest.hit.yPix));
    std::copy(src.hit.ePix.begin(), src.hit.ePix.end(), back_inserter(dest.hit.ePix));
}

//TODO: all barycenters and variances are calculated using the raw values of positive pixels, not rounded to nearest integer - this may not be what you want
void computeHitParameters(const track_t &hit, Float_t *hitParam){

    double xSum   = 0;
    double ySum   = 0;
    double wSum   = 0;
    int    eSum   = 0;
    int    nPix   = 0;
    for(unsigned int i=0;i<hit.xPix.size();++i){
        const double &ePix = hit.ePix[i]; 
        if(ePix>0){
            xSum += hit.xPix[i]*ePix;
            ySum += hit.yPix[i]*ePix;
            wSum += ePix;
        }
        eSum += hit.ePixInt[i];
        ++nPix;
    }
    float xBary = xSum/wSum;
    float yBary = ySum/wSum;

    double x2Sum = 0;
    double y2Sum = 0;


    for(unsigned int i=0;i<hit.xPix.size();++i){
        const double &ePix = hit.ePix[i];
            if(ePix>0){
                double dx = (hit.xPix[i] - xBary);
                double dy = (hit.yPix[i] - yBary);
                x2Sum   += dx*dx*ePix;
                y2Sum   += dy*dy*ePix;

            }
    }

    // variances at 45 degrees. Used to compute R.
    double Abis =atan(1.0);
    double xy1Sumbis =0; //xy1Sum is in the axis where dx*dy>0 and xy2Sum where dx*dy<0
    double xy2Sumbis =0;
    for(unsigned int i=0;i<hit.xPix.size();++i){
        const double &ePix = hit.ePix[i];
            if(ePix>0){
                double dx = (hit.xPix[i] - xBary);
                double dy = (hit.yPix[i] - yBary);
                double a = atan(dy/dx);
                xy1Sumbis +=(dx*dx+dy*dy)*cos(a-Abis)*cos(a-Abis)*ePix;
                xy2Sumbis +=(dx*dx+dy*dy)*sin(a-Abis)*sin(a-Abis)*ePix; //perpendicular to that, -45 degrees.
            }
    }

    // variances on main axis
    double A =atan(y2Sum/x2Sum);
    double xy1Sum =0; //xy1Sum is in the axis where dx*dy>0 and xy2Sum where dx*dy<0
    double xy2Sum =0;
    double xy1SumWidth=0;
    double xy2SumWidth=0;
    for(unsigned int i=0;i<hit.xPix.size();++i){
        const double &ePix = hit.ePix[i];
            if(ePix>0){
                double dx = (hit.xPix[i] - xBary);
                double dy = (hit.yPix[i] - yBary);
                double a = atan(dy/dx);
                xy1Sum +=(dx*dx+dy*dy)*cos(a-A)*cos(a-A)*ePix;
                xy2Sum +=(dx*dx+dy*dy)*sin(a-A)*sin(a-A)*ePix;
                xy1SumWidth +=pow((dx*dx+dy*dy)*cos(a-A)*cos(a-A),0.5);
                xy2SumWidth +=pow((dx*dx+dy*dy)*sin(a-A)*sin(a-A),0.5);
            }
    }


    float xVar = x2Sum/wSum;
    float yVar = y2Sum/wSum;
    float xy1Var = xy1Sum/wSum;
    float xy2Var = xy2Sum/wSum;
    float xy1Varbis = xy1Sumbis/wSum;
    float xy2Varbis = xy2Sumbis/wSum;
    float xy1Width =xy1SumWidth/hit.xPix.size();
    float xy2Width =xy2SumWidth/hit.xPix.size();
    float alpha = A;
    double sum=(xy1Varbis+xy2Varbis+xVar+yVar)/4;
    double r = ((xy1Varbis-sum)*(xy1Varbis-sum)+(xy2Varbis-sum)*(xy2Varbis-sum)+(xVar-sum)*(xVar-sum)+(yVar-sum)*(yVar-sum))/4;
    float R= r;
    hitParam[0] = eSum;
    hitParam[1] = nPix;
    hitParam[2] = xBary;
    hitParam[3] = yBary;
    hitParam[4] = xVar;
    hitParam[5] = yVar;
    hitParam[6] = wSum;
    hitParam[7] = xy1Var;
    hitParam[8] = xy2Var;
    hitParam[9] = alpha*180/3.1415;
    hitParam[10] = R;
    hitParam[11] = xy1Width;
    hitParam[12] = xy2Width;
}

void setClusterFlags(vector<hitTreeEntry_t*> &tracks, const int nX, const int nY, const eMaskType maskType, const int* maskArray, const float* distanceArray, const float* bleedXArray, const float* bleedYArray, const float* clusterDistArray){
    int nclusters = tracks.size();
    for (int iHit=0; iHit<nclusters; iHit++) {
        
        hitTreeEntry_t &evt = *tracks[iHit];
        evt.hit.flag = 0;//clear flags
        int nhit = evt.hit.xPix.size();
        for (int iPix=0; iPix<nhit; iPix++) {
            int x = evt.hit.xPix[iPix];
            int y = evt.hit.yPix[iPix];
            int i = x + nX*y;
            evt.hit.flag |= maskArray[i];
            if( maskType == eComputeMask || maskType==ePartialMask){ // If internal mask has been generated, use the distances
                evt.hit.bleedX   = min(evt.hit.bleedX, bleedXArray[i]);
                evt.hit.bleedY   = min(evt.hit.bleedY, bleedYArray[i]);
                evt.hit.distance = min(evt.hit.distance, distanceArray[i]);
                evt.hit.clusterDist = min(evt.hit.clusterDist, clusterDistArray[i]);
            }
        }
    }
}


//second pass of masking: this runs after clustering
void maskClusters(vector<hitTreeEntry_t*> &tracks, const double* ePixArray, const int* ePixIntArray, const int ohdu, const ccdGeom_t &ccdGeom, const long totpix, const int nX, const int nY, int* maskArray, const double maxVal, float* clusterDistArray){
    gConfig &gc = gConfig::getInstance();

    const int clusterThreshold = gc.clusterThr(); //e- threshold for us to draw a mini-halo around a cluster
    const int clusterCut = gc.clusterCut(); //mini-halo radius

    //const int looseClusterRadius = 20; //radius for loose clusters

    const double fullWellThreshold = min(0.9*maxVal, (double) gc.fullWellThr()); //pixel threshold for full-well clusters

    const int singlePixThreshold = gc.singlePixThr(); //e- threshold for dropping single-pixel clusters
    const int horzThreshold = gc.horzThr(); //e- threshold for dropping horizontal (skipper-CTI) clusters
    const bool dropHorz = gc.isDropHorz(ohdu);
    
    int nhits = tracks.size();
    //first loop: full-well event mask
    for (int iHit=0; iHit<nhits; iHit++) {

        hitTreeEntry_t &evt = *tracks[iHit];
        if (evt.ohdu!=ohdu) continue; //only look at clusters in this HDU
        int nPix = evt.hitParam[1];//cast back to int
        const int xLowEdge = (ccdGeom.npre+1)/ccdGeom.bincol;//first active pixel is x=NPRESCAN+1
        //full-well cluster mask
        int fullPix=0;
        //count full-well pixels
        for (int iPix=0; iPix<nPix; iPix++) {
            if (evt.hit.xPix[iPix]<xLowEdge) {//ignore a cluster if it extends to the left of the active area (this excludes the blob of charge that often forms at the edge of the active area)
                fullPix = 0;
                break;
            }
            if (evt.hit.ePix[iPix] > fullWellThreshold) {
                fullPix++;
            }
        }
        if (fullPix >= 10) {//mask all pixels to the right of every full-well pixel in this cluster
            for (int iPix=0; iPix<nPix; iPix++) if (evt.hit.ePix[iPix] > 0.8*fullWellThreshold) {
                int hitX = evt.hit.xPix[iPix];
                int hitY = evt.hit.yPix[iPix];
                int minY = max(0, hitY-2); //mask starting 2 rows below
                int maxY = min(nY-1, hitY+2); //mask up to 2 rows above
                for (int y = minY; y<=maxY; y++) {
                    int startX = hitX;
                    if (gc.fullWellMaskLeft()) startX = 0;
                    for (int x = startX; x<nX; x++) {
                        if (maskArray[y*nX + x] & kFullWell) {//if we run into a pixel that's already masked, everything to the right will also be masked, so this row is done
                            break;
                        }
                        maskArray[y*nX + x] |= kFullWell;
                    }
                }
            }
        }
    }

    

    // second loop: SR hit
    const int SRlengthOfBox = 30;
    const int SRminSep = 5;
    const int SRhitsPerBox = 3;
    const int SRmultToMask = 2;
    const int SRmaxNeighborHitsPerBox = 1;
    const int SRignoreMask = kImageTooNoisy+kBadPix+kBadCol+kExtendedBleed+kFullWell;

    for (int yi = 1; yi<nY-1; yi++) {
        // we need to count pixels in a sliding window, we do this efficiently by incrementing/decrementing as the right/left window edges pass over pixels
        int hitsBelow=0, hitsAbove=0;
        std::set<int> inWindow;
        for (int xi = 0; xi<SRlengthOfBox; xi++) {
            const int i = yi*nX + xi;
            if (ePixIntArray[i]>0 && (maskArray[i] & SRignoreMask)==0) inWindow.insert(i);
            if (ePixIntArray[i-nX]>0) hitsBelow++;
            if (ePixIntArray[i+nX]>0) hitsAbove++;
        }
        int maskEnd = 0; //we persist this so we always know where the previous mask (if any) ended
        for (int xi = 0; xi<nX-SRlengthOfBox; xi++) {
            const int i = yi*nX + xi;
            const int maskedHits = inWindow.size();
            if (maskedHits>=SRhitsPerBox && hitsBelow<=SRmaxNeighborHitsPerBox && hitsAbove<=SRmaxNeighborHitsPerBox) {
                const int first = *inWindow.begin();
                const int last = *inWindow.rbegin();
                if (last-first >= SRminSep) {
                    //printf("boom h%d y%d x%d %d %d %d %d\n",ext,yi,xi,maskedHits,inWindow.size(),*inWindow.begin(),*inWindow.rbegin());
                    int maskStart = max(xi - SRmultToMask*SRlengthOfBox, maskEnd);
                    maskEnd = min(xi + (SRmultToMask+1)*SRlengthOfBox, nX);
                    for (int xMask = maskStart; xMask<maskEnd; xMask++) {
                        maskArray[yi*nX + xMask] |= kOverscanEvent;
                    }
                }
            }

            //update the counts
            if (ePixIntArray[i]>0 && (maskArray[i] & SRignoreMask)==0) inWindow.erase(i);
            if (ePixIntArray[i-nX]>0) hitsBelow--;
            if (ePixIntArray[i+nX]>0) hitsAbove--;
            if (ePixIntArray[i+SRlengthOfBox]>0 && (maskArray[i+SRlengthOfBox] & SRignoreMask)==0) inWindow.insert(i+SRlengthOfBox);
            if (ePixIntArray[i-nX + SRlengthOfBox]>0) hitsBelow++;
            if (ePixIntArray[i+nX + SRlengthOfBox]>0) hitsAbove++;
        }
    }

    // const int smallClusterMaskBits = kNearBigEvent+kBadPix+kBadCol+kFullWell+kCrossTalk+kOverscanEvent+kInBleedZone+kExtendedBleed; //low-E cluster does not apply to pixels that contain any of these mask bits

    // third loop: cluster mask
    for (int iHit=0; iHit<nhits; iHit++) {
        //we assume clusterThreshold is not larger than 5 and clusterCut is at least 4 (otherwise kCluster does not work as intended)

        hitTreeEntry_t &evt = *tracks[iHit];
        if (evt.ohdu!=ohdu) continue; //only look at clusters in this HDU

        int nEle = evt.hitParam[0];//cast back to int
        int nPix = evt.hitParam[1];//cast back to int
        if (nEle>=clusterThreshold) {
            //printf("%d %d %d\n", evt.ohdu, nPix, nEle);

            //loop through pixels in the cluster - check the mask bits and fill the neighbor set
            std::set<std::pair<int,int>> neighborSet; //set of pixels in, or neighboring, this cluster
            for (int iPix=0; iPix<nPix; iPix++) {
                int x = evt.hit.xPix[iPix];
                int y = evt.hit.yPix[iPix];
                neighborSet.insert(make_pair(x,y));
                // for (int dy=-1; dy<=1; dy++) {
                //     if (y+dy<0 || y+dy>=nY) continue; //bounds check
                //     for (int dx=-1; dx<=1; dx++) {
                //         if (x+dx<0 || x+dx>=nX) continue; //bounds check
                //         if (!gc.getUseDiagonalPix() && (dx*dy!=0)) continue; //if we are not clustering diagonal pixels, they should not count as neighbors
                //         neighborSet.insert(make_pair(x+dx,y+dy));
                //     }
                // }
            }

            //loop again through pixels in this cluster, mask nearby pixels
            for (int iPix=0; iPix<nPix; iPix++) {
                int x = evt.hit.xPix[iPix];
                int y = evt.hit.yPix[iPix];
                // bool badSmallCluster = (maskArray[x+nX*y] & smallClusterMaskBits);
                bool badSmallCluster = false;

                
                //printf("%d %d %f\n", evt.hit.xPix[iPix], evt.hit.yPix[iPix], evt.hit.ePix[iPix]);

                // i%nX is x position
                // i/nX is y position
                const int rStart = max(0,    x - clusterCut/ccdGeom.bincol-1);
                const int rEnd   = min(nX-1, x + clusterCut/ccdGeom.bincol+1);
                const int sStart = max(0,    y - clusterCut/ccdGeom.binrow-1);
                const int sEnd   = min(nY-1, y + clusterCut/ccdGeom.binrow+1);

                // cout << rStart << " " << rEnd << " " << sStart << " " << sEnd << endl;
                for (int r = rStart; r <= rEnd; ++r)
                {
                    float dX = (r - (x)) * ccdGeom.bincol;
                    for (int s = sStart; s <= sEnd; ++s)
                    {
                        float dY = (s - (y)) * ccdGeom.binrow;
                        float distXY = sqrt(pow(dX,2)+pow(dY,2));
                        if (!badSmallCluster && distXY<clusterCut && nEle!=gc.clusterIgnore() && neighborSet.count(make_pair(r,s))==0){ //so we draw circles and not rectangles; don't mask out the cluster we're using or the pixels neighboring it
                            maskArray[s*nX+r]  |= kSmallCluster;
                            if (distXY<clusterDistArray[s*nX+r]) {
                                clusterDistArray[s*nX+r] = distXY;
                            }
                        }
                        if (distXY<4 && nEle>=5) { //fixed radius=4 and threshold=5 for kCluster, which is used to set cuts
                            maskArray[s*nX+r]  |= kCluster;
                        }
                    }
                }
            }
        }
    }

    // fourth loop: cluster shape cuts
    for (int iHit=0; iHit<nhits; iHit++) {
        hitTreeEntry_t &evt = *tracks[iHit];
        if (evt.ohdu!=ohdu) continue; //only look at clusters in this HDU

        int nEle = evt.hitParam[0];//cast back to int
        int nPix = evt.hitParam[1];//cast back to int

        if (nEle<=1) continue; // 1e events are always fine
        if (evt.hit.yMax != evt.hit.yMin) continue; //clusters with nonzero vertical extent are always fine

        if ((nPix==1 && nEle>=singlePixThreshold) || (nPix==nEle && (dropHorz || nEle>=horzThreshold))) {
            //loop through pixels in the cluster - check the mask bits and fill the neighbor set
            for (int iPix=0; iPix<nPix; iPix++) {
                int x = evt.hit.xPix[iPix];
                int y = evt.hit.yPix[iPix];
                maskArray[y*nX + x] |= kClusterShape;
            }
        }
    }
}

void initHitTree(TTree &hitSumm, hitTreeEntry_t &evt , int &runID, int &expoStart, int &LTANAME){
    hitSumm.Branch("runID",    &runID,    "runID/I");
    hitSumm.Branch("LTANAME",    &LTANAME,    "LTANAME/I");
    hitSumm.Branch("ohdu",     &(evt.ohdu),     "ohdu/I");
    hitSumm.Branch("expoStart",     &expoStart,     "expoStart/I");

    hitSumm.Branch("nSat", &(evt.hit.nSat), "nSat/I");
    hitSumm.Branch("flag", &(evt.hit.flag), "flag/I");
    hitSumm.Branch("xMin", &(evt.hit.xMin), "xMin/I");
    hitSumm.Branch("xMax", &(evt.hit.xMax), "xMax/I");
    hitSumm.Branch("yMin", &(evt.hit.yMin), "yMin/I");
    hitSumm.Branch("yMax", &(evt.hit.yMax), "yMax/I");
    hitSumm.Branch("distance", &(evt.hit.distance), "distance/F");
    hitSumm.Branch("bleedX", &(evt.hit.bleedX), "bleedX/F");
    hitSumm.Branch("bleedY", &(evt.hit.bleedY), "bleedY/F");
    hitSumm.Branch("clusterDist", &(evt.hit.clusterDist), "clusterDist/F");

    for(int n=0;n<gNExtraTNtupleVars;++n){
        hitSumm.Branch(gExtraTNtupleVars[n],  &(evt.hitParam[n]),  (string(gExtraTNtupleVars[n])+"/F").c_str());
    }

    hitSumm.Branch("nSavedPix", &(evt.nSavedPix), "nSavedPix/I");
    hitSumm.Branch("xPix", &(evt.hit.xPix[0]), "xPix[nSavedPix]/I");
    hitSumm.Branch("yPix", &(evt.hit.yPix[0]), "yPix[nSavedPix]/I");
    hitSumm.Branch("ePix", &(evt.hit.ePix[0]), "ePix[nSavedPix]/F");

}

void refreshTreeAddresses(TTree &hitSumm, hitTreeEntry_t &evt)
{
    hitSumm.SetBranchAddress("xPix", &(evt.hit.xPix[0]));
    hitSumm.SetBranchAddress("yPix", &(evt.hit.yPix[0]));
    hitSumm.SetBranchAddress("ePix", &(evt.hit.ePix[0]));
}


std::string trim(const std::string& str, const std::string& whitespace = " \t\'"){ // removes leading and trailing spaces
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos) return ""; // no content
    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;
    return str.substr(strBegin, strRange);
}

int getExpoInfoFromHeader(fitsfile *fptr, Int_t &tShut, Int_t &tExpo){
    int status = 0;   /*  CFITSIO status value MUST be initialized to zero!  */

    int hdutype;
    char keyValue[1024] = "";
    char comment[1024]  = "";

    Int_t tDate = -1;
    tShut = -1;
    tExpo = -1;
    // Get general data from ext=0 (hdu=1) header
    fits_movabs_hdu(fptr, 1, &hdutype, &status);
    if(status!=0) return -2;

    fits_read_keyword(fptr, "DATE", keyValue, comment, &status);
    if(status==0){ // key exist
        std::tm tm = {};
        strptime(trim(keyValue).c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
        time_t t = std::mktime(&tm);
        tDate = t;	    
    }

    fits_read_keyword(fptr, "UTSHUT", keyValue, comment, &status);
    if(status==0){ // key exist
        std::tm tm = {};
        strptime(trim(keyValue).c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
        time_t t = std::mktime(&tm);
        tShut = t;	    
    }
    tExpo = tDate - tShut;

    return 0;
}

inline bool isInteger(const char *c, int &n){
    if(strlen(c)==0) return false;
    string s = trim(c);
    if(((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false ;
    std::istringstream iss(s);
    float f;
    iss >> noskipws >> f; // noskipws considers leading whitespace invalid
    if( (iss.eof() && !iss.fail()) == false ) return false; // it's not a number
    if(f != int(f) ) return false; // it's not integer

    n = int(f);
    return true;
}

int getCcdGeometryInfoFromHeader(fitsfile *fptr, ccdGeom_t &ccdGeom){
    int status = 0;   /*  CFITSIO status value MUST be initialized to zero!  */

    //int  hdutype;
    char keyValue[1024] = "";
    char comment[1024]  = "";

    fits_read_keyword(fptr, "CCDNCOL", keyValue, comment, &status);
    if(status==0){ // key exist
        if(isInteger(keyValue,ccdGeom.ncol)==false) return -1; 
    }
    status = 0;

    fits_read_keyword(fptr, "CCDNROW", keyValue, comment, &status);
    if(status==0){ // key exist
        if(isInteger(keyValue,ccdGeom.nrow)==false) return -1;
    }
    status = 0;

    fits_read_keyword(fptr, "CCDNPRES", keyValue, comment, &status);
    if(status==0){ // key exist
        if(isInteger(keyValue,ccdGeom.npre)==false) return -1;
    }
    status = 0;

    fits_read_keyword(fptr, "NBINROW", keyValue, comment, &status);
    if(status==0){ // key exist
        if(isInteger(keyValue,ccdGeom.binrow)==false) return -1;
    }else{
        ccdGeom.binrow=1;
    }
    status = 0;

    fits_read_keyword(fptr, "NBINCOL", keyValue, comment, &status);
    if(status==0){ // key exist
        if(isInteger(keyValue,ccdGeom.bincol)==false) return -1;
    }else{
        ccdGeom.bincol=1;
    }
    status = 0;
    status = 0;

    return 0;
}


int searchForTracks(TFile *outF, vector<hitTreeEntry_t*> &tracks, const double* ePixArray, const int* ePixIntArray, const int ohdu, const ccdGeom_t &ccdGeom, const long totpix, const int nX, const int nY){
    const int kNVars = gNBaseTNtupleVars + gNExtraTNtupleVars;
    Float_t *ntVars = new Float_t[kNVars];

    bool * pixUsed = new bool[totpix](); //initialized to false
    bool useDiagonalPix = gConfig::getInstance().getUseDiagonalPix();

    queue<int> clustering_queue;

    for(long i=0;i<totpix;++i){

        if (!pixUsed[i] && ePixIntArray[i]>0) {
            clustering_queue.push(i);
        } else {
            pixUsed[i] = true;
            continue;
        }

        hitTreeEntry_t * newtrack = new hitTreeEntry_t(gNExtraTNtupleVars);
        tracks.push_back(newtrack);
        //newtrack->runID = runID;
        newtrack->ohdu = ohdu;
        //newtrack->expoStart = expoStart;
        track_t &hit = newtrack->hit;
        hit.id = tracks.size()-1;

        while (!clustering_queue.empty()) {
            int newhit = clustering_queue.front();
            clustering_queue.pop();
            if (!pixUsed[newhit] && ePixIntArray[newhit]>0) {
                int hitX = newhit%nX;
                int hitY = newhit/nX;

                if( hit.xMin > hitX ) hit.xMin = hitX;
                if( hit.xMax < hitX ) hit.xMax = hitX;
                if( hit.yMin > hitY ) hit.yMin = hitY;
                if( hit.yMax < hitY ) hit.yMax = hitY;

                hit.fill(hitX, hitY, ePixArray[newhit], ePixIntArray[newhit]);

                //enqueue next hits
                if(hitX>0)    clustering_queue.push(newhit-1); //West
                if(hitY>0)    clustering_queue.push(newhit-nX); //South
                if(hitX<nX-1) clustering_queue.push(newhit+1); //East
                if(hitY<nY-1) clustering_queue.push(newhit+nX); //North
                if(useDiagonalPix){
                    if(hitX>0    && hitY<nY-1) clustering_queue.push(newhit+nX-1); //North-West
                    if(hitX>0    && hitY>0)    clustering_queue.push(newhit-nX-1); //South-West
                    if(hitX<nX-1 && hitY>0)    clustering_queue.push(newhit-nX+1); //South-East
                    if(hitX<nX-1 && hitY<nY-1) clustering_queue.push(newhit+nX+1); //North-East
                }
            }
            pixUsed[newhit]=true;
        }
        computeHitParameters( hit, newtrack->hitParam );
        if(gVerbosity){
            if(i%1000 == 0) showProgress(i,totpix);
        }
    }
    delete[] ntVars;

    if(gVerbosity){
        showProgress(1,1);
    }

    return 0;
}


bool readCardValue(fitsfile  *infptr, const char *keyName, double &value){

    int status = 0;
    char record[1024] = "";
    fits_read_card(infptr, keyName, record, &status);
    if(status==KEY_NO_EXIST){
        status=0;
        return false;
    }
    else{
        string sRec(record);
        size_t tPosEq = sRec.find("=");
        size_t tPosSl = sRec.find("/");
        string sVal(sRec.substr(tPosEq+1, tPosSl-tPosEq-1));
        std::replace( sVal.begin(), sVal.end(), '\'', ' ');
        istringstream recISS( sVal );
        recISS >> value;
        return true;
    }

}

bool readStrCard(fitsfile  *infptr, const char *keyName, char *value){
    int status = 0;
    fits_read_key(infptr, TSTRING, keyName, value, NULL, &status);
    return (status!=KEY_NO_EXIST);
}

bool readIntCard(fitsfile  *infptr, const char *keyName, int &value){
    int status = 0;
    int oldValue = value;
    fits_read_key(infptr, TINT, keyName, &value, NULL, &status);
    if (status!=0) value = oldValue; //if any error (no key, or value empty/invalid), restore the original value
    return (status!=KEY_NO_EXIST);
}

int readMask(const char* maskName, vector <int*> &masks, const vector<int> &singleHdu){
    int status = 0;
    int nhdu = 0;
    double nulval = 0.;
    int anynul = 0;


    fitsfile  *infptr; /* FITS file pointers defined in fitsio.h */
    fits_open_file(&infptr, maskName, READONLY, &status); /* Open the input file */
    if (status != 0) return(status);
    fits_get_num_hdus(infptr, &nhdu, &status);
    if (status != 0) return(status);

    /* check the extensions to process*/
    for(unsigned int i=0;i<singleHdu.size();++i){
        if(singleHdu[i] > nhdu){
            fits_close_file(infptr,  &status);
            cerr << red << "\nError: the file does not have the required HDU!\n\n" << normal;
            return -1000;
        }
    }

    vector<int> useHdu(singleHdu);
    if(singleHdu.size() == 0){
        for(int i=0;i<nhdu;++i){
            useHdu.push_back(i+1);
        }
    }
    const unsigned int nUseHdu=useHdu.size();

    for (unsigned int eN=0; eN<nUseHdu; ++eN)  /* Main loop through each extension */
    {
        const int n = useHdu[eN];

        /* get input image dimensions and total number of pixels in image */
        int hdutype, bitpix, naxis = 0;
        long naxes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        fits_movabs_hdu(infptr, n, &hdutype, &status);
        for (int i = 0; i < 9; ++i) naxes[i] = 1;
        fits_get_img_param(infptr, 9, &bitpix, &naxis, naxes, &status);
        long totpix = naxes[0] * naxes[1];

        /* Don't try to process data if the hdu is empty */    
        if (hdutype != IMAGE_HDU || naxis == 0 || totpix == 0){
            masks.push_back(0);
            continue;
        }

        int* maskArray = new int[totpix];

        /* Open the input file */
        fits_movabs_hdu(infptr, n, &hdutype, &status);
        if (status != 0) return(status);

        /* Read the images as doubles, regardless of actual datatype. */
        long fpixel[2]={1,1};
        long lpixel[2]={naxes[0],naxes[1]};
        long inc[2]={1,1};
        fits_read_subset(infptr, TINT, fpixel, lpixel, inc, &nulval, maskArray, &anynul, &status);
        if (status != 0){
            fits_report_error(stderr, status);
            return(status);
        }
        masks.push_back(maskArray);
    }
    fits_close_file(infptr,  &status);
    return status;
}

void writeConfigTree(TFile *outF){

    gConfig &gc = gConfig::getInstance();
    gCal &gcal = gCal::getInstance();

    outF->cd();
    TTree configTree("config","config");

    Float_t kCal[101];
    for(int i=0;i<=100;++i){
        kCal[i] = gcal.isValid()?gcal.getExtCal(i):gc.getExtCal(i);
    }
    configTree.Branch("cal", &kCal, "cal[101]/F");

    Bool_t kSaveTracks  = gc.getSaveTracks();
    configTree.Branch("saveTracks", &kSaveTracks, "saveTracks/B");

    TString kTrackCuts  = gc.getTracksCuts();
    configTree.Branch("trackCuts", &kTrackCuts);

    Int_t kHalo  = gc.haloCut();
    configTree.Branch("halo", &kHalo, "halo/I");

    Int_t kEdge  = gc.edgeCut();
    configTree.Branch("edge", &kEdge, "edge/I");

    Int_t kBleedX  = gc.bleedXCut();
    configTree.Branch("bleedX", &kBleedX, "bleedX/I");

    Int_t kBleedY  = gc.bleedYCut();
    configTree.Branch("bleedY", &kBleedY, "bleedY/I");

    Int_t kCluster  = gc.clusterCut();
    configTree.Branch("clusterDist", &kCluster, "clusterDist/I");


    configTree.Fill();
    configTree.Write();
}


int copyStructure(const char* inF, const char *outF){

    fitsfile  *outfptr; /* FITS file pointers defined in fitsio.h */
    fitsfile *infptr;   /* FITS file pointers defined in fitsio.h */

    int status = 0;
    // int single = 0;

    int hdutype, bitpix, naxis = 0, nkeys;
    int nhdu = 0;
    long naxes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    long totpix = 0;

    fits_open_file(&infptr, inF, READONLY, &status); /* Open the input file */
    if (status != 0) return(status);

    fits_get_num_hdus(infptr, &nhdu, &status);  

    fits_create_file(&outfptr, outF, &status);/* Create the output file */
    if (status != 0) return(status);


    for (int n=1; n<=nhdu; ++n)  /* Main loop through each extension */
    { 
        /* get image dimensions and total number of pixels in image */
        fits_movabs_hdu(infptr, n, &hdutype, &status);
        for (int i = 0; i < 9; ++i) naxes[i] = 1;
        fits_get_img_param(infptr, 9, &bitpix, &naxis, naxes, &status);
        totpix = naxes[0] * naxes[1] * naxes[2] * naxes[3] * naxes[4] * naxes[5] * naxes[6] * naxes[7] * naxes[8];

        if (hdutype != IMAGE_HDU || naxis == 0 || totpix == 0){
            /* just copy tables and null images */
            fits_copy_hdu(infptr, outfptr, 0, &status);
            if (status != 0) return(status);
        }
        else{
            fits_create_img(outfptr, LONG_IMG, naxis, naxes, &status);//image of signed int values
            if (status != 0) return(status);

            /* copy the header keywords */
            fits_get_hdrspace(infptr, &nkeys, NULL, &status); 
            for (int i = 1; i <= nkeys; ++i){
                char card[FLEN_CARD];
                fits_read_record(infptr, i, card, &status);
                if (fits_get_keyclass(card) > TYP_CMPRS_KEY) fits_write_record(outfptr, card, &status);
            }

        }
    }

    fits_close_file(infptr, &status);
    fits_close_file(outfptr,  &status);

    return status;
}

double computeMedian( std::vector<double> v, const size_t N) {
    if (N%2==0) {
        nth_element(v.begin(), v.begin()+(N/2), v.end());
        nth_element(v.begin(), v.begin()+(N/2)+1, v.end());
        return (v[N/2] + v[N/2 + 1])/2;
    } else {
        nth_element(v.begin(), v.begin()+(N/2), v.end());
        return v[N/2];
    }
}

double computeMAD2( std::vector<double> v, const size_t &N, const double &median){//used to compute row and column MADs
    std::vector<double> vTmpMad(v);
    for(size_t i=0; i<N; ++i) vTmpMad[i] = abs(v[i]-median);
    return computeMedian(vTmpMad, N);
}

double computeMAD( double* v, const size_t &N, const double &median){//original code used to compute the image MAD, this does not seem to follow the standard definition of MAD
    std::vector<double> vTmpMad(v, v+N);
    //std::transform(v, v+N, vTmpMad.begin(), [median](double vElmt){return abs(vElmt-median);}); // doesn't work on gcc 4.4
    for(size_t i=0; i<N; ++i) vTmpMad[i] = abs(v[i]-median);
    size_t centerElmt = N / 2;
    nth_element(vTmpMad.begin(), vTmpMad.begin()+centerElmt, vTmpMad.end());
    auto mad = vTmpMad[centerElmt];
    return mad;
}

double getMaxVal(const double* ePixArray, const long totpix, const int nX, const int nY, const ccdGeom_t &ccdGeom){ //find maximum pixel value in the active area - used for full-well mask
    // Compute boundaries of active area
    gConfig &gc = gConfig::getInstance();
    const int xLowEdge = (ccdGeom.npre+1)/ccdGeom.bincol;//first active pixel is x=NPRESCAN+1
    const int xHiEdge  = min( (ccdGeom.npre + ccdGeom.ncol)/ccdGeom.bincol, nX);
    int yLowEdge = 1;//first active pixel is y=1
    int yHiEdge  = min( ccdGeom.nrow/ccdGeom.binrow, nY);
    if (!gc.firstRowPrescan()) { //if the first row is active
        yLowEdge--;
        yHiEdge--;
    }

    bool isValid = false;
    double maxVal;
    for (int iy = yLowEdge; iy <= yHiEdge; iy++) {
        for (int ix = xLowEdge; ix <= xHiEdge; ix++) {
            if (isValid) {
                maxVal = max(maxVal, ePixArray[ix+nX*iy]);
            } else {
                maxVal = ePixArray[ix+nX*iy];
                isValid = true;
            }
        }
    }
    return maxVal;
}

// first pass of masking: this runs before clustering
int buildMask(const int* ePixIntArray, const int ext, const long totpix, const int nX, const int nY, int* maskArray, float* distanceArray, float* azimuthArray, float* bleedXArray, float* bleedYArray, int* crossArray, const int nBleedThr, const ccdGeom_t &ccdGeom=ccdGeom_t()){

    gConfig &gc = gConfig::getInstance();
    gCal &gcal = gCal::getInstance();

    int nEdgeMask     = gc.edgeCut();
    if (nEdgeMask==-1){nEdgeMask=gc.haloCut();} //follow halo mask if edgeCut not specified in config file

    // Compute boundaries for edge-mask - pixels outside the boundaries are masked, pixels on the boundary are not masked
    const int xLowEdge = (ccdGeom.npre+1 + nEdgeMask)/ccdGeom.bincol;//first active pixel is x=NPRESCAN+1
    const int xHiEdge  = min( (ccdGeom.npre + ccdGeom.ncol)/ccdGeom.bincol, nX) - nEdgeMask/ccdGeom.bincol;
    int yLowEdge = 1 + nEdgeMask/ccdGeom.binrow;//first active pixel is y=1
    int yHiEdge  = min( ccdGeom.nrow/ccdGeom.binrow, nY) - nEdgeMask/ccdGeom.binrow;
    if (!gc.firstRowPrescan()) { //if the first row is active
        yLowEdge--;
        yHiEdge--;
    }


    //first loop: mask crosstalk, edges, neighbors, bad pix+cols
    for (int i = 0; i < totpix; ++i)
    {

        /* Edges mask */
        const int xi = i%nX;
        const int yi = i/nX;
        if(xi<xLowEdge-1) maskArray[i] |= kNearEdge;
        if(xi>xHiEdge+1)  maskArray[i] |= kNearEdge;
        if(yi<yLowEdge-1) maskArray[i] |= kNearEdge;
        if(yi>yHiEdge+1)  maskArray[i] |= kNearEdge;

        if(crossArray!=0) maskArray[i] |= crossArray[i];

        if (gc.isBadCol(ext, xi)) maskArray[i] |= kBadCol;
        if (gc.isBadPix(ext, xi, yi)) maskArray[i] |= kBadPix;

        if(ePixIntArray[i]>0){
            maskArray[i] |= kNoEmptyMask; //If you have more than 0.5 electrons, you are 2
            if(xi>0)    maskArray[i-1]  |= kHasNeighbour; // Explanation below...
            if(xi<nX-1) maskArray[i+1]  |= kHasNeighbour; // More..
            if(yi>0)    maskArray[i-nX] |= kHasNeighbour; // Almost there...
            if(yi<nY-1) maskArray[i+nX] |= kHasNeighbour; // These four lines set the pixels that are neighbour of a kNoEmptyMask pixel.
            if(gc.getUseDiagonalPix()){
                if(xi>0    && yi<nY-1) maskArray[i+nX-1]  |= kHasNeighbour; // Explanation below...
                if(xi>0    && yi>0)    maskArray[i-nX-1]  |= kHasNeighbour; // More..
                if(xi<nX-1 && yi>0)    maskArray[i-nX+1] |= kHasNeighbour; // Almost there...
                if(xi<nX-1 && yi<nY-1) maskArray[i+nX+1] |= kHasNeighbour; // These four lines set the pixels that are neighbour of a kNoEmptyMask pixel.
            }
        }
    }

    //second loop: mask bleed and halo
    //std::map<int,std::pair<int,int>> bleedX_counts;
    for (int i = 0; i < totpix; ++i)
    {
        const int xi = i%nX;
        const int yi = i/nX;
        if(ePixIntArray[i]>nBleedThr){
            // bleed zone... If the pixel has more than 101 electrons... Mask 50 pixels on right and 50 up

            bool antibleed=false; //by default set to false. If not, will calculate antibleeding.
            bool scaleBleedYAsTraps = true; //if false, cut distance scales as a distance in the active area; if true, cut distance scales as a readout time

            // mask a line above a high-charge pixel
            // this type of CTI is dominated by traps and is dependent on time not the number of transfers
            int bleedYCutDistance = scaleBleedYAsTraps?gc.bleedYCut()*ccdGeom.bincol:gc.bleedYCut()/ccdGeom.binrow; //in units of image pixels
            int minBleedDY = 1;
            if (antibleed) minBleedDY = 1 - bleedYCutDistance;
            for (int j = minBleedDY; j < bleedYCutDistance; ++j)
            {
                if (yi+j<0) continue; //don't go outside the image
                if (yi+j==nY) break; //don't go outside the image

                if (j==0) { //special case for antibleeding only: a high-charge pixel is considered to have 0 bleed distance
                    bleedYArray[i+j*nX] = 0;
                    continue;
                }

                if (j>0) maskArray[i+j*nX] |= kInBleedZone; //we don't mask antibleeding

                float bleedDistance = j*ccdGeom.binrow; //physical distance on the CCD
                if (gc.isBleedCol(ext, xi)) bleedDistance /= 2; // if this is a bleed col, scale accordingly (we want bleedY to represent the value of the bleed distance cut that would mask this pixel)
                if (abs(bleedYArray[i+j*nX]) > abs(bleedDistance)) bleedYArray[i+j*nX] = bleedDistance; //for antibleeding, we consider negative and positive distances the same
            }

            if (gc.isBleedCol(ext, xi)) { //double the bleed distance for bleed columns
                for (int j = bleedYCutDistance; j < 2*bleedYCutDistance; ++j)
                {
                    if (yi+j==nY) break; //don't go outside the image

                    maskArray[i+j*nX] |= kExtendedBleed;
                    float bleedDistance = j*ccdGeom.binrow/2; //physical distance on the CCD
                    if (abs(bleedYArray[i+j*nX]) > abs(bleedDistance)) bleedYArray[i+j*nX] = bleedDistance; //for antibleeding, we consider negative and positive distances the same
                }
            }

            // mask a line to the right of a high-charge pixel
            int bleedXCutDistance = gc.bleedXCut()/ccdGeom.bincol; //in units of image pixels
            if (gc.isPastBleedXEdge(ext, xi)) { //if we are past the "bleed edge" for this quadrant, the X-bleed extends forever
                bleedXCutDistance = INT_MAX;
            }
            int minBleedDX = 1;
            if (antibleed) minBleedDX = 1 - bleedXCutDistance;
            for (int j = minBleedDX; j < bleedXCutDistance; ++j)
            {
                if (xi+j<0) continue; //don't go outside the image
                if (xi+j==nX) break; //don't go outside the image

                if (j==0) { //special case for antibleeding only: a high-charge pixel is considered to have 0 bleed distance
                    bleedXArray[i+j] = 0;
                    continue;
                }

                if (j>0) maskArray[i+j] |= kInBleedZone; //we don't mask antibleeding

                float bleedDistance = j*ccdGeom.bincol; //physical distance on the CCD
                if (abs(bleedXArray[i+j]) > abs(bleedDistance)) bleedXArray[i+j] = bleedDistance;

                /*
                //this code is for identifying bleed rows: with the low dark current in the new CCDs, there is not enough charge in the bleed region for this cut to work, so this code is commented
                int ePix_here = (int) (imgArray[i+j]/kCal - gc.epixCut() + 1);//rounds down to the number of electrons
                int mask_here = maskArray[i+j];
                if (ePix_here<6 && !(mask_here & (kInBleedZone+kCrossTalk+kNearEdge+kOverscanEvent))) {//no more than 5e-, not already masked
                //update counts
                if (bleedX_counts.count(yi)==0) bleedX_counts[yi] = std::make_pair(0,0);
                bleedX_counts[yi].first++;
                if (ePix_here>0) {//at least 1e-
                //printf("found bleed: x=%d y=%d pix=%f %d\n", xi+j, yi, imgArray[i+j], ePix_here);
                bleedX_counts[yi].second += ePix_here;
                }
                }
                */
            }

            //halo mask

            // i%nX is x position
            // i/nX is y position
            const int rStart = max(0,    i%nX - gc.haloCut()/ccdGeom.bincol-1);
            const int rEnd   = min(nX-1, i%nX + gc.haloCut()/ccdGeom.bincol+1);
            const int sStart = max(0,    i/nX - gc.haloCut()/ccdGeom.binrow-1);
            const int sEnd   = min(nY-1, i/nX + gc.haloCut()/ccdGeom.binrow+1);


            // cout << rStart << " " << rEnd << " " << sStart << " " << sEnd << endl;
            for (int r = rStart; r <= rEnd; ++r)
            {
                float dX = max((abs(r - (i%nX)) - 1) * ccdGeom.bincol + 1, 0); //round down if binning, so we mask a superpixel even if only part of it is inside the radius
                for (int s = sStart; s <= sEnd; ++s)
                {
                    float dY = max((abs(s - (i/nX)) - 1) * ccdGeom.binrow + 1, 0);
                    float dXY = sqrt(pow(dX,2) + pow(dY,2));
                    if (dXY < gc.haloCut()){ //so we draw circles and not rectangles
                        maskArray[s*nX+r]  |= kNearBigEvent;
                        if (dXY<distanceArray[s*nX+r]) {
                            distanceArray[s*nX+r] = dXY;
                            if (dXY>0) {
                                azimuthArray[s*nX+r] = atan2(dY, dX);
                            }
                        }
                    }
                }
            }
        } 
    }

    /*
    //this code is for identifying bleed rows: with the low dark current in the new CCDs, there is not enough charge in the bleed region for this cut to work, so this code is commented
    std::vector<double> bleedX_rates;
    std::map<int, std::pair<int,int>>::iterator bleedX_it = bleedX_counts.begin();
    int nZero = 0;
    int nNonzero = 0;
    while (bleedX_it != bleedX_counts.end()) {
//printf("%d %d %d %f\n", bleedX_it->first, bleedX_it->second.first, bleedX_it->second.second, ((float) bleedX_it->second.second)/bleedX_it->second.first);
if (bleedX_it->second.second==0) {
nZero++;
} else {
nNonzero++;
}

bleedX_rates.push_back( ((float) bleedX_it->second.second)/bleedX_it->second.first);
bleedX_it++;
}
double bleedX_median = computeMedian(bleedX_rates, bleedX_rates.size());
double bleedX_MAD = computeMAD2(bleedX_rates, bleedX_rates.size(), bleedX_median);
printf("%d %d %f %f\n", nZero, nNonzero, bleedX_median, bleedX_MAD);
*/

return 0;
}

int findBig(double* imgArray, const double kCal, const long totpix, int* crossArray)
{
    gConfig &gc = gConfig::getInstance();
    // const double kSeedThr   = kCal*gc.epixCut();

    for (int i = 0; i < totpix; ++i)
    {
        if(imgArray[i]>(0.5+gc.crosstalkThr())*kCal){ /* If we're over 700 electrons, mark this for crosstalk */
            crossArray[i] = kCrossTalk;
            if (i+1 < totpix) crossArray[i+1] = kCrossTalk; //also mask the next pixel
        }
    }
    return 0;
}

int writeMask(const char *maskfName, const int ext, const long totpix, int* maskArray) {
    int status = 0;
    /* Open the output file */
    fitsfile  *maskfptr; /* FITS file pointers defined in fitsio.h */
    fits_open_file(&maskfptr, maskfName, READWRITE, &status);
    if (status != 0) return(status);

    int hdutype;
    fits_movabs_hdu(maskfptr, ext, &hdutype, &status);
    fits_write_img(maskfptr, TINT, 1, totpix, maskArray, &status);
    fits_close_file(maskfptr,  &status);

    return status;
}

int computeImage(const vector<string> &inFileList, const eMaskType maskType, const char *maskName, const char *outFile, const vector<int> &singleHdu, const int &nBleedThr){
    int status = 0;
    double nulval = 0.;
    int anynul = 0;

    gConfig &gc = gConfig::getInstance();
    gCal &gcal = gCal::getInstance();
    cout << "nBleedThr " << nBleedThr << endl; 

    // Mask handling
    vector <int*> masks;
    // read the provided mask file, if there is one
    if(maskType==eExternalMask || maskType==ePartialMask){
        status = readMask(maskName, masks, singleHdu);
        if (status!=0) return status;
    } 
    // End of mask handling

    int nhdu = 0;
    const unsigned int nFiles  = inFileList.size();

    TFile outRootFile(outFile, "RECREATE");

    writeConfigTree(&outRootFile);

    // these parameters are no longer used
    Int_t xSize     = -1;
    Int_t ySize     = -1;
    Int_t tShut     = -1;
    Int_t tExpo     = -1;
    Double_t tTemp  = -1;

    Int_t tRunID    = -1;
    // the LTA name is not guaranteed to be an integer, but we assume this anyway
    Int_t tLTANAME    = -1;

    // Pixel tree variables
    Int_t xPix    = -1;
    Int_t yPix    = -1;
    Double_t ePix = -1;
    Int_t maskVal = -1;
    Float_t distanceVal = -1;
    Float_t azimuthVal = -1;
    Int_t bleedXVal = -1;
    Int_t bleedYVal = -1;
    Float_t clusterDistVal = -1;
    Int_t ohdu    = -1;
    TTree calPixTree("calPixTree","calPixTree");
    calPixTree.Branch("x",    &xPix, "x/I");
    calPixTree.Branch("y",    &yPix, "y/I");
    calPixTree.Branch("ePix", &ePix, "ePix/D");
    calPixTree.Branch("mask", &maskVal, "mask/I");

    calPixTree.Branch("distance", &distanceVal, "distance/F");
    calPixTree.Branch("azimuth", &azimuthVal, "azimuth/F");
    calPixTree.Branch("bleedX", &bleedXVal, "bleedX/I");
    calPixTree.Branch("bleedY", &bleedYVal, "bleedY/I");
    calPixTree.Branch("clusterDist", &clusterDistVal, "clusterDist/F");
    calPixTree.Branch("ohdu", &ohdu, "ohdu/I");
    calPixTree.Branch("RUNID", &tRunID, "RUNID/I");
    calPixTree.Branch("LTANAME", &tLTANAME, "LTANAME/I");

    const bool kSaveTracks = gc.getSaveTracks();
    outRootFile.cd();
    TTree hitSumm("hitSumm","hitSumm");
    hitTreeEntry_t evt(gNExtraTNtupleVars);
    initHitTree(hitSumm, evt, tRunID, tExpo,tLTANAME);

    TTree hitSummAux("hitSummAux","hitSummAux");
    if(kSaveTracks){
        initHitTree(hitSummAux, evt, tRunID, tExpo,tLTANAME);
        hitSummAux.SetCircular(1);
    }


    for(unsigned int fn=0; fn < nFiles; ++fn){

        // Initialize mask fits file if option to compute it is provided
        string maskfName="deleteMe";
        if(maskType==eComputeMask || maskType==ePartialMask){
            const int nameStart     = std::max((int)(inFileList[fn].rfind("/")+1), 0);
            const int nameEnd       = std::max((int)(inFileList[fn].rfind(".fits")+1), 0);
            const int extraProcName = (inFileList[fn].substr(nameStart,5)=="proc_") ? 5 : 0;
            string outFileBaseName  = inFileList[fn].substr(nameStart+extraProcName, nameEnd-nameStart-extraProcName-1);
            maskfName               = "mask_"+outFileBaseName+".fits";
            cout << bold << "\nWill save computed mask to:\n"  << normal;
            cout << "\t" << maskfName << endl << endl;
            if(fileExist(maskfName.c_str())) deleteFile(maskfName.c_str());
            copyStructure(inFileList[fn].c_str(), maskfName.c_str());
        }


        fitsfile  *infptr; /* FITS file pointers defined in fitsio.h */
        fits_open_file(&infptr, inFileList[fn].c_str(), READONLY, &status); /* Open the input file */
        if (status != 0) return(status);
        fits_get_num_hdus(infptr, &nhdu, &status);
        if (status != 0) return(status);

        getExpoInfoFromHeader(infptr, tShut, tExpo);
        readIntCard(infptr, "RUNID", tRunID);
        //readStrCard(infptr, "LTANAME", ltaName);
        //tLTANAME = atoi(ltaName);
        readIntCard(infptr, "LTANAME", tLTANAME);
        readCardValue(infptr, "TEMPER", tTemp);
        fitsHeaderToTree(infptr, &outRootFile);

        // load the correct set of pixel and column masks, and print it
        gc.selectLta(tLTANAME);
        cout << "loading masks for LTA " << tLTANAME << endl << endl;
        // gc.printMask();

        /* check the extensions to process*/
        for(unsigned int i=0;i<singleHdu.size();++i){
            if(singleHdu[i] > nhdu){
                fits_close_file(infptr,  &status);
                cerr << red << "\nError: the file does not have the required HDU!\n\n" << normal;
                return -1000;
            }
        }

        vector<int> useHdu(singleHdu);
        if(singleHdu.size() == 0){
            for(int i=0;i<nhdu;++i){
                useHdu.push_back(i+1);
            }
        }
        const unsigned int nUseHdu=useHdu.size();


        //first loop through HDUs: mark crosstalk pixels
        int *crossArray=0;
        if(gc.crosstalkThr()>0 && (maskType == eComputeMask || maskType==ePartialMask)){ // If internal mask is going to be generated we search for cross-talk candidate pixels to include them in the mask
            for (unsigned int eN=0; eN<nUseHdu; ++eN)  /* Loop through all extension to build cross-talk mask */
            {
                const int n = useHdu[eN];

                /* get input image dimensions and total number of pixels in image */
                int hdutype, bitpix, naxis = 0;
                long naxes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
                fits_movabs_hdu(infptr, n, &hdutype, &status);
                for (int i = 0; i < 9; ++i) naxes[i] = 1;
                fits_get_img_param(infptr, 9, &bitpix, &naxis, naxes, &status);
                long totpix = naxes[0] * naxes[1];

                /* Don't try to process data if the hdu is empty */    
                if (hdutype != IMAGE_HDU || naxis == 0 || totpix == 0){
                    continue;
                }

                if(crossArray==NULL){
                    crossArray = (new int[totpix]);
                    std::fill_n(crossArray, totpix, 0);
                }

                double* outArray = new double[totpix];

                /* Open the input file */
                fits_movabs_hdu(infptr, n, &hdutype, &status);
                if (status != 0) return(status);

                /* Read the images as doubles, regardless of actual datatype. */
                long fpixel[2]={1,1};
                long lpixel[2]={naxes[0],naxes[1]};
                long inc[2]={1,1};

                fits_read_subset(infptr, TDOUBLE, fpixel, lpixel, inc, &nulval, outArray, &anynul, &status);
                if (status != 0) return(status);

                ohdu = n;
                readIntCard(infptr, "OHDU", ohdu);
                const double kCal = gcal.isValid()?gcal.getExtCal(ohdu):gc.getExtCal(ohdu);
                findBig(outArray, kCal, totpix, crossArray);
                delete[] outArray;
            }
        }

        for (unsigned int eN=0; eN<nUseHdu; ++eN)  /* Main loop through each extension */
        {

            const int n = useHdu[eN];

            /* get input image dimensions and total number of pixels in image */
            int hdutype, bitpix, naxis = 0;
            long naxes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
            fits_movabs_hdu(infptr, n, &hdutype, &status);
            for (int i = 0; i < 9; ++i) naxes[i] = 1;
            fits_get_img_param(infptr, 9, &bitpix, &naxis, naxes, &status);
            long totpix = naxes[0] * naxes[1];

            /* Don't try to process data if the hdu is empty */    
            if (hdutype != IMAGE_HDU || naxis == 0 || totpix == 0){
                continue;
            }

            double* ePixArray = new double[totpix]; //electrons per pixel, not rounded

            /* Open the input file */
            fits_movabs_hdu(infptr, n, &hdutype, &status);
            if (status != 0) return(status);
            if(xSize<naxes[0]) xSize = naxes[0];
            if(ySize<naxes[1]) ySize = naxes[1];

            /* Read the images as doubles, regardless of actual datatype. */
            long fpixel[2]={1,1};
            long lpixel[2]={naxes[0],naxes[1]};
            long inc[2]={1,1};

            // read the raw ADUs into ePixArray
            fits_read_subset(infptr, TDOUBLE, fpixel, lpixel, inc, &nulval, ePixArray, &anynul, &status);
            if (status != 0) return(status);

            ohdu = n;
            readIntCard(infptr, "OHDU", ohdu);
            double expoStart = 0;
            readCardValue(infptr, "EXPSTART", expoStart);

            const double kCal = gcal.isValid()?gcal.getExtCal(ohdu):gc.getExtCal(ohdu);
            for (int i=0; i<totpix; ++i) { //scale ePixArray to get electrons
                ePixArray[i] /= kCal;
            }

            int* ePixIntArray = new int[totpix]; //electrons per pixel, after applying 1e- threshold or rounding
            for (int i=0; i<totpix; ++i) {
                if (ePixArray[i]<gc.epixCut()) { //threshold to have 1 or more e-
                    ePixIntArray[i] = 0;
                } else if (ePixArray[i]<gc.epixCut2()) { //threshold to have 2 or more e-
                    ePixIntArray[i] = 1;
                } else {
                    ePixIntArray[i] = floor(ePixArray[i] - gc.epixCutN()) + 1;
                }
            }

            ccdGeom_t ccdGeom;
            getCcdGeometryInfoFromHeader(infptr, ccdGeom);

            // guess whether the CCD is split in H and/or V by comparing the image size to the CCD size
            // if the CCD size > the image size in one dimension we assume that we are splitting the CCD in that dimension
            if(ccdGeom.ncol+ccdGeom.npre+1 > naxes[0]*ccdGeom.bincol) ccdGeom.ncol /= 2;
            if (gc.firstRowPrescan()) {
                if (ccdGeom.nrow > (naxes[1]-1)*ccdGeom.binrow) ccdGeom.nrow /= 2;
            } else {
                if (ccdGeom.nrow > naxes[1]*ccdGeom.binrow) ccdGeom.nrow /= 2;
            }

            // if dimensions were specified in the config file, they take precedence
            if (gc.getCCDNCOL()>0) {
                ccdGeom.ncol = gc.getCCDNCOL();
            }
            if (gc.getCCDNROW()>0) {
                ccdGeom.nrow = gc.getCCDNROW();
            }

            if(gVerbosity){
                cout << "\nProcessing runID " << tRunID << " ohdu " << ohdu << " -> cal = " << kCal << ":\n";
            }

            int* maskArray = 0;
            switch(maskType)
            {
                case eNoMask:      //initialize mask to 0
                case eComputeMask: //initialize mask to 0
                    maskArray = new int[totpix];
                    std::fill_n(maskArray, totpix, 0);
                    break;

                case eExternalMask:
                case ePartialMask:
                    maskArray = masks[eN];
                    break;

            }

            float* distanceArray = new float[totpix];
            std::fill_n(distanceArray, totpix, gc.haloCut()+1);
            float* azimuthArray = new float[totpix]; //don't bother initialize this, values are invalid unless distance>0
            float* clusterDistArray = new float[totpix];
            std::fill_n(clusterDistArray, totpix, gc.clusterCut()+1);
            float* bleedXArray = new float[totpix];
            std::fill_n(bleedXArray, totpix, gc.bleedXCut()+1);
            float* bleedYArray = new float[totpix];
            std::fill_n(bleedYArray, totpix, gc.bleedYCut()+1);

            double maxVal;
            bool goodQuad = true; //process this quad

            if (maskType==eComputeMask || maskType==ePartialMask){

                if (gc.dropNoise()) {
                    const double kNoise = gcal.getExtNoise(ohdu);
                    for (int yi = 0; yi < naxes[1]; yi++) {
                        int nNoisyPix = 0;
                        int nZeroPix = 0;
                        for (int xi = 0; xi < naxes[0]; xi++) {
                            double ePix = ePixArray[xi+naxes[0]*yi];
                            if (ePix < 1-2*kNoise) {
                                if (abs(ePix) > 2*kNoise) {
                                    //printf("%d %d %f\n", xi, yi, ePix);
                                    nNoisyPix++;
                                } else {
                                    nZeroPix++;
                                }
                            }
                        }
                        if (nNoisyPix > 0.06*naxes[0] || nZeroPix < 0.5*naxes[0]) {
                            //printf("%d %d %d\n", yi, nNoisyPix, nZeroPix);
                            for (int xi = 0; xi < naxes[0]; xi++) {
                                maskArray[xi+naxes[0]*yi] |= kImageTooNoisy;
                            }
                        }
                    }
                } else {
                    // compute the median absolute value
                    const double mad = computeMAD( ePixArray, totpix, 0);
                    if( mad > 0.6){
                        for (int i = 0; i < totpix; ++i) maskArray[i] |= kImageTooNoisy;
                        cout << "\n\nWARNING: Image in ohdu="<< ohdu << " is too noisy. MAD=" << mad <<  " is above threshold";
                        goodQuad = false;
                    }
                }

                maxVal = getMaxVal(ePixArray, totpix, naxes[0], naxes[1], ccdGeom);
                //printf("max val in hdu %d: %f e-\n",ohdu, maxVal);

                if (goodQuad) {
                    buildMask(ePixIntArray, ohdu, totpix, naxes[0], naxes[1], maskArray, distanceArray, azimuthArray, bleedXArray, bleedYArray, crossArray, nBleedThr, ccdGeom);
                }
            }

            vector<hitTreeEntry_t*> tracks;
            if (goodQuad) {
                searchForTracks(&outRootFile, tracks, ePixArray, ePixIntArray, ohdu, ccdGeom, totpix, naxes[0], naxes[1]);


                //apply pixel masks that depend on clustering
                if (maskType==eComputeMask || maskType==ePartialMask) {
                    maskClusters(tracks, ePixArray, ePixIntArray, ohdu, ccdGeom, totpix, naxes[0], naxes[1], maskArray, maxVal, clusterDistArray);
                }

                //update cluster flags
                setClusterFlags(tracks, naxes[0], naxes[1], maskType, maskArray, distanceArray, bleedXArray, bleedYArray, clusterDistArray);
            }


            const string kTrackCuts = gc.getTracksCuts().c_str();

            int nClusters = tracks.size();
            for (int i=0; i<nClusters; i++) {
                copyTrack(*tracks[i], evt);
                evt.nSavedPix = 0;
                if(kSaveTracks){
                    hitSummAux.Fill();
                    if( hitSummAux.GetEntries(kTrackCuts.c_str()) == 1 ){ //only save the pixel lists if this track passes the cut string defined in the XML config
                        evt.nSavedPix = evt.hit.xPix.size();
                        refreshTreeAddresses(hitSumm, evt);
                    }
                }
                hitSumm.Fill();
            }


            //fill the pixel tree
            for (int i = 0; i < totpix; ++i){
                xPix    = i%(naxes[0]);
                yPix    = i/(naxes[0]);
                ePix    = ePixArray[i];
                maskVal = (maskArray==0)? 0 : maskArray[i];
                distanceVal = (distanceArray==0)? 0 : distanceArray[i];
                azimuthVal = (azimuthArray==0)? 0 : azimuthArray[i];
                bleedXVal = (bleedXArray==0)? 0 : bleedXArray[i];
                bleedYVal = (bleedYArray==0)? 0 : bleedYArray[i];
                clusterDistVal = (clusterDistArray==0)? 0 : clusterDistArray[i];
                calPixTree.Fill(); 
            }

            /* clean up */
            if(maskType == eComputeMask || maskType==ePartialMask) {
                if (gc.getMaskSmall()){
                    for (int i = 0; i < totpix; ++i){
                        maskArray[i] &= 255; //deletes every mask above 128
                    }
                }
                writeMask(maskfName.c_str(), ohdu, totpix, maskArray);
                delete[] maskArray;
            }

            delete[] ePixArray;
            delete[] ePixIntArray;
            delete[] distanceArray;
            delete[] azimuthArray;
            delete[] clusterDistArray;
            delete[] bleedXArray;
            delete[] bleedYArray;
        }

        if(crossArray!=NULL) delete[] crossArray;


        /* Close the input file */
        fits_close_file(infptr,  &status);   

    }
    outRootFile.cd();

    calPixTree.Write();
    hitSumm.Write();

    TTree imgParTree("imgParTree", "imgParTree");
    imgParTree.Branch("xSize",    &xSize, "xSize/I");
    imgParTree.Branch("ySize",    &ySize, "ySize/I");
    imgParTree.Branch("tShut",    &tShut, "tShut/I");
    imgParTree.Branch("tExpo",    &tExpo, "tExpo/I");
    imgParTree.Branch("tRunID",   &tRunID, "tRunID/I");
    imgParTree.Branch("tTemp",    &tTemp, "tTemp/D");
    imgParTree.Fill();
    imgParTree.Write();
    outRootFile.Close();

    return status;
}


void checkArch(){
    if(sizeof(float)*CHAR_BIT!=32 || sizeof(double)*CHAR_BIT!=64){
        cout << red;
        cout << "\n ========================================================================================\n";
        cout << "   WARNING: the size of the float and double variables is non-standard in this computer.\n";
        cout << "   The program may malfunction or produce incorrect results\n";
        cout << " ========================================================================================\n";
        cout << normal;
    }
}


int processCommandLineArgs(const int argc, char *argv[], 
        vector<int> &singleHdu, vector<string> &inFileList, eMaskType &maskType, string &maskFile, string &outFile, int &nBleedThr){

    if(argc == 1) return 1;
    inFileList.clear();
    singleHdu.clear();
    nBleedThr=100; //default mask values
    bool outFileFlag = false;
    maskType = eNoMask;
    gCal &gcal = gCal::getInstance();
    gConfig &gc = gConfig::getInstance();
    int opt=0;
    while ( (opt = getopt(argc, argv, "am:p:o:s:c:C:b:t:qhH?")) != -1) {
        switch (opt) {
            case 'a':
                if(maskType==eNoMask){
                    maskFile = "** compute mask **";
                    maskType = eComputeMask;
                }
                else{
                    cerr << red << "\nError, can not set more than one mask file!\n\n" << normal;
                    return 2;
                }
                break;
            case 'm':
                if(maskType==eNoMask){
                    maskFile = optarg;
                    maskType = eExternalMask;
                }
                else{
                    cerr << red << "\nError, can not set more than one mask file!\n\n" << normal;
                    return 2;
                }
                break;
            case 'p':
                if(maskType==eNoMask){
                    maskFile = optarg;
                    maskType = ePartialMask;
                }
                else{
                    cerr << red << "\nError, can not set more than one mask file!\n\n" << normal;
                    return 2;
                }
                break;
            case 'o':
                if(!outFileFlag){
                    outFile = optarg;
                    outFileFlag = true;
                }
                else{
                    cerr << red << "\nError, can not set more than one output file!\n\n" << normal;
                    return 2;
                }
                break;
            case 's':
                singleHdu.push_back(atoi(optarg));
                break;
            case 'c':
                if(gc.readConfFile(optarg) == false){
                    return 1;
                }
                break;
            case 'C':
                if(gcal.readConfFile(optarg) == false){
                    return 1;
                }
                break;
            case 'b': //setting big event mask aka halo
                gc.setHaloCut(atoi(optarg));
                break;
            case 't': //setting big event mask aka halo
                //	nBleedThrVector.push_back(atoi(optarg));
                nBleedThr=atoi(optarg);
                break;
            case 'q':
                gVerbosity = 0;
                break;
            case 'h':
            case 'H':



            default: /* '?' */
                return 1;
        }
    }

    if(maskType==eNoMask){
        cerr << yellow << "\nMask filename missing. Will use empty mask\n" << normal;
        maskFile = "";
    }

    for(int i=optind; i<argc; ++i){
        inFileList.push_back(argv[i]);
        if(!fileExist(argv[i])){
            cout << red << "\nError reading input file: " << argv[i] <<"\nThe file doesn't exist!\n\n" << normal;
            return 1;
        }
    }

    if(inFileList.size()==0){
        cerr << red << "Error: no input file(s) provided!\n\n" << normal;
        return 1;
    }

    if(!outFileFlag){
        cerr << yellow << "\nWarning: output filename missing. Will use default:\n" << normal;
        const int nameStart     = std::max((int)(inFileList[0].rfind("/")+1), 0);
        const int nameEnd       = std::max((int)(inFileList[0].rfind(".fits")+1), 0);
        const int extraProcName = (inFileList[0].substr(nameStart,5)=="proc_") ? 5 : 0;
        string outFileBaseName  = inFileList[0].substr(nameStart+extraProcName, nameEnd-nameStart-extraProcName-1);
        outFile = "hits_"+outFileBaseName+".root";
        cout <<  "\t" << outFile << endl << endl;
    }

    return 0;
}

#include <iostream>
#include <sstream>
#include <locale>
#include <iomanip>
int main(int argc, char *argv[])
{
    checkArch(); //Check the size of the double and float variables.

    time_t start,end;
    double dif;
    time (&start);

    eMaskType maskType;
    string maskFile;
    string outFile;
    vector<string> inFileList;
    vector<int> singleHdu;
    int nBleedThr;

    int returnCode = processCommandLineArgs( argc, argv, singleHdu, inFileList, maskType, maskFile, outFile, nBleedThr);
    if(returnCode!=0){
        if(returnCode == 1) printCopyHelp(argv[0],true);
        if(returnCode == 2) printCopyHelp(argv[0]);
        return returnCode;
    }

    gCal &gcal = gCal::getInstance();
    if(gVerbosity && gcal.isValid()){
        cout << "\nCalibration file: " << gcal.filename() << endl;
        gcal.printVariables();
    }

    /* Create configuration singleton and read configuration file */
    gConfig &gc = gConfig::getInstance();
    if (!gc.isValid()) {
        if(gc.readConfFile("extractConfig.xml") == false){
            return 1;
        }
    }
    if(gVerbosity){
        cout << "\nConfig file: " << gc.filename() << endl;
        gc.printVariables();
    }

    if(gVerbosity){
        cout << bold << "\nWill read the following files:\n" << normal;
        for(unsigned int i=0; i<inFileList.size();++i){
            cout << "\t" << inFileList[i] << endl;
            cout << "\t" << maskFile << endl;

        }
        if(singleHdu.size()>0){
            cout << bold << "\nAnd the following extension:" << normal << endl << "\t";
            for(unsigned int i=0; i<singleHdu.size();++i) cout << singleHdu[i] << ",";
            cout << "\b " << endl;
        }
        cout << bold << "\nThe output will be saved in the file:\n\t" << normal << outFile << endl;
    }

    int status = computeImage(inFileList, maskType, maskFile.c_str(), outFile.c_str(), singleHdu, nBleedThr);
    if (status != 0){ 
        fits_report_error(stderr, status);
        return status;
    }

    /* Report */
    time (&end);
    dif = difftime (end,start);
    if(gVerbosity) cout << green << "\nAll done!\n" << bold << "-> It took me " << dif << " seconds to do it!\n\n" << normal;

    return status;
}
