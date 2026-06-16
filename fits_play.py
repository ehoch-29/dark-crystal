from astropy.io import fits

with fits.open('smpl_500samp_200rows_2_2_1.fits') as hdul:
    hdul.info()
    for i, hdu in enumerate(hdul):
        print(f"\n--- HDU {i}: {type(hdu).__name__} ---")
        print(f"  Header keys: {list(hdu.header.keys())[:10]}")
        if hdu.data is not None:
            print(f"  Data shape:  {hdu.data.shape}")
            print(f"  Data dtype:  {hdu.data.dtype}")
