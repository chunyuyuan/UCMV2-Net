# UCM-NetV2

This is the official code repository for "UCM-NetV2 and BNN-UCM-NetV2: Efficient and Accurate Deep Learning Models for Skin Lesion Segmentation " 

**Prepare the dataset.**

- The ISIC2017 and ISIC2018 datasets,which follows the same as UltraLight-VM-UNet , can be found here {[One Drive]}. 
- {[ISIC2017](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/EcNlo_WWGVJBtAf_F6AFrrABmSAJi9J3TlHwrnsW8ccOPw?e=NRScI8)}
- {[ISIC2018](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/EZr0xZCCRJZBv9HW4dYWr94BlHmD_9dg6k1QbnzMmH6Asw?e=RVDise)}
-  (PS: you can use above link to directly download the dataset and put them in the data folder or use UltraLight-VM-UNet code to generate the division of dataset)


**Train the UCMV2-UNet.**
```
cd UCMV2-UNet
```

```
# ISIC2017 
python trainucm.py  --data 'ISIC2017' 

```
```
# ISIC2018 
python trainucm.py  --data 'ISIC2018' 

```



** Citation **
```
@article{yuan2024ucm,
  title={UCM-NetV2 and BNN-UCM-NetV2: Efficient and Accurate Deep Learning Models for Skin Lesion Segmentation on Mobile Devices},
  author={Yuan, Chunyu and Zhao, Dongfang and Agaian, Sos S},
  year={2024},
  publisher={Preprints}
}

```
**Acknowledgement.**

Some code,dataset generaation, data loading, model trainning , are cited from UltraLight-VM-UNet {[UltraLight-VM-UNet ](https://github.com/wurenkai/UltraLight-VM-UNet)}.

We thank their wonderful works!
