
call "D:\tools\Andorid\Sdk\ndk\20.0.5594570\ndk-build.cmd"



adb push .\libs\arm64-v8a\androidOCL_64 /data/data/test/

adb shell chmod 777 /data/data/test/androidOCL_64
adb shell /data/data/test/androidOCL_64