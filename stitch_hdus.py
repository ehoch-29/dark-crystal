import numpy as np
from astropy.io import fits

def main(fn=None, out_fn=None, overwrite=False, verbose=False):

    fn = '../data/images/proc_2025_01_22_full_image300.fits' if fn is None else fn
    out_fn = fn.split('.fits')[0] + '_stitched.fits' if out_fn is None else out_fn

    stitched_fits = stitch_hdus(fn, verbose)

    stitched_fits.writeto(out_fn, overwrite=overwrite)

    print(f"stitched {fn} --> {out_fn}")


def stitch_hdus(fn, verbose=False):
    """
    Returns a full CCD image: stitches the four individual HDU data from each amplifier into
    a single image with the proper orientation, removing overscan regions.

    """

    hdul = fits.open(fn)
    assert len(hdul) == 4, f"expected 4 HDUs but found {len(hdul)}"

    # get info from the header
    hdr = hdul[0].header
    # physical number of rows and columns
    nrow, ncol = int(hdr['CCDNROW']), int(hdr['CCDNCOL'])
    # physical number of rows and columns in each hdu (i.e. data.shape after we remove overscan)
    xdim, ydim = ncol // 2, nrow // 2

    # remove overscan
    data = [
        hdu.data[:ydim, :xdim] for hdu in hdul
    ]
    
    # now we populate the full image, starting in the corners,
    #   since the hdu doesn't necessary scan the entire region

    full_image = np.full((nrow, ncol), np.nan)  # size of the full CCD array including overscan regions
        # (physical CCD array is (1024, 6144) i.e. if we remove the overscan)

    # top left corner: original data needs to be flipped vertically
    full_image[ydim:, :xdim] = data[0][::-1,:]
    # top right corner: original data needs to be flipped horizontally and vertically
    full_image[ydim:, xdim:] = data[1][::-1,::-1]
    # bottom left corner: original data already has the correct orientation
    full_image[:ydim, :xdim] = data[2]
    # bottom right corner: original data needs to be flipped horizontally
    full_image[:ydim, xdim:] = data[3][:,::-1]

    # are there any unpopulated pixels in the final image?
    nnan = np.sum(np.isnan(full_image))
    if verbose and nnan > 0:
        print(f"warning: {nnan} unfilled pixels in the stitched image")

    hdu = fits.PrimaryHDU(data=full_image)
    return hdu


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default=None, type=str,
                            help='FITS file to stitch')
    parser.add_argument('-n', '--name', default=None, type=str,
                            help='where to save stitched file')
    parser.add_argument('-o', '--overwrite', default=False, type=bool,
                            help='whether to overwrite existing stitched file')
    parser.add_argument('-v', '--verbose', default=False, type=bool,
                            help='verbose')
    args = parser.parse_args()
    main(args.filename, args.name, args.overwrite, args.verbose)