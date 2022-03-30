<template>
    <div id="wrapper">
        <h1>Style Transfer</h1>
        <div class="container"> <!--主内容-->
             <!--section 内容/风格/输出：section-header + section-main + 按钮-->
            <div class="content-section">
                <h3 class="section-header">Content Image</h3>
                <div class="section-main"> 
                    <div class="subsection">
                        <p>upload one content image</p>
                        <!--显示图像 todo-->
                        <div class="center-container"> <!--center-container 中心画布-->
                            <img v-if="contentSrc"
                            :src="contentSrc"
                            alt="">
                        </div>
                    </div>
                </div>
                <!--上传content图像文件，响应显示-->
                <form id="upload-file"
                        method="post"
                        enctype="multipart/form-data"> <!--enctype 不对字符编码,在使用包含文件上传控件的表单时，必须使用该值。 为什么不用action属性？-->
                    <label for="imageUpload" class="btn">Choose...</label>
                    <input type="file"
                            name="file"
                            @change="contentUpload"
                            id="imageUpload"
                            accept=".png, .jpg, .jpeg">
                </form>
                
            </div>
            
            <div class="style-section">
                <h3 class="section-header">Style Image</h3>
                <div class="section-main">
                    <div class="subsection"> <!--section-main分成几个subsection-->
                        <p>choose one style image</p>
                        <div class="center-container">
                            <!-- 风格小图-->
                            <div class="image-grid">
                                <div v-for="image in styleImages"
                                    :key="image.id"
                                    class="image-item">
                                    <image-item :src="image.src"
                                        :id="image.id"
                                        :selected="selectedId"
                                        @clicked="onSelectStyle"></image-item>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="subsection">
                        <p>change blend weights</p>
                        <div class="center-container">
                        </div>
                    </div>
                </div>
                
                <button class="btn"
                        :disabled="submitDisable"
                        @click="submit">
                    <span>Submit</span>
                </button>
            </div>

            <div class="result-section">
                <h3 class="section-header">Output</h3>
                <div class="section-main">
                    <div class="subsection">
                        <p>...</p>
                        <!--显示图像 todo-->
                        <div class="center-container">
                            <img v-if="resultSrc"
                            :src="resultSrc"
                            alt="">
                        </div>
                    </div>
                </div>
            </div>
            
        </div>
    </div>
    
    
</template>

<script>
// import { defineComponent } from '@vue/composition-api'
// 子组件
import ImageItem from "./ImageItem.vue";

import axios from "axios";

// 创建axios实例
const axiosStyle =
  process.env.NODE_ENV === "development"
    ? axios.create({ baseURL: "http://localhost:5001" })
    : axios.create({ baseURL: "http://localhost:5001" });

export default ({
    name: "LadingPage",

    components: {
        ImageItem
    },

    data(){
        return {
            // 可选择的style
            styleImages: [
                { id: "0", src: require("@/assets/thumbs/bird.jpg") },
                { id: "1", src: require("@/assets/thumbs/candy_style10.jpg") },
                { id: "2", src: require("@/assets/thumbs/canvas.jpg") },
                { id: "3", src: require("@/assets/thumbs/cat.jpg") },
                { id: "4", src: require("@/assets/thumbs/composition_vii.jpg") },
                { id: "5", src: require("@/assets/thumbs/dr_gache.jpg") },
                { id: "6", src: require("@/assets/thumbs/edtaonisl.jpg") },
                { id: "7", src: require("@/assets/thumbs/empire.jpg") },
                { id: "8", src: require("@/assets/thumbs/escher_sphere.jpg") },
                { id: "9", src: require("@/assets/thumbs/eternal_style2.jpg") },
            ],
            // selectedId: 0, // 最终选中的styleId
            selectedId: [],

            sessionId: "",

            // content
            // 需要contentData
            contentSrc: "",

            // style
            // styleIds: [0], // test 写死
            
            // model
            modelId: 0, // 暂定

            // result
            resultSrc: "",

            // submit
            submitDisable: true
        }
    },
    
    mounted: function() {
        this.sessionId =
            "_" +
            Math.random()
                .toString(36)
                .substr(2, 9);
        
        // 测试服务器连接，执行get请求
        axiosStyle.get("/")
            .then(response => {
                console.log(response.data);
            })
            .catch(function(error) {
                console.log("style server erro");
                console.log(error);
            });
            
    },

    methods: {
        onSelectStyle(id) { // 点击一个小图时
            // this.selectedId = id; // 点击后就会把this.selectedId改成当前选中的

            let index = this.selectedId.indexOf(id);
            if (index > -1){ // 若已经选中，则取消选中
                this.selectedId.splice(index, 1);
            }else{ // 若未选中，则选中
                this.selectedId.push(id);
            }     

            // 判断是否可提交
            if(this.selectedId.length == 0){
                this.submitDisable = true;
            }else{
                this.submitDisable = false;
            }
        },

        submit(){
            // 构建数据
            var styleData = new FormData();
            styleData.append("id", this.sessionId);
            // styleData.append("styleIds", this.styleIds);
            styleData.append("styleIds", this.selectedId);
            styleData.append("contentData", this.contentSrc); // todo
            styleData.append("modelId", this.modelId);

            // 创建请求
            axiosStyle({
                url: "/stylize-with-data",
                method: "POST",
                data: styleData,
                headers: {
                    "Content-Type": "multipart/form-data"
                }
            }).then(response => {
                // 处理返回数据
                this.resultSrc = response.data;
                
            });

        },
        contentUpload(e) {
            var files = e.target.files || e.dataTransfer.files;
            // 选择文件后 或 拖拽上传; *.files是FileList对象
            if (!files.length) return;

            var reader = new FileReader(); // 创建FileReader对象

            reader.onload = e => { 
                /*
                var canvas = document.querySelector("#canvas");
                var ctx = canvas.getContext("2d");

                var w = canvas.width;
                var h = canvas.height;
                ctx.clearRect(0, 0, w, h);

                var img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0, w, h);
                };
                img.src = e.target.result;
                */
                this.contentSrc = e.target.result;
            };
            reader.readAsDataURL(files[0]);  // base64编码
            e.target.value = "";
            this.userContent = true;
        },
        /*
        onSelectStyle(id) {

        }
        */
    }
    
})
</script>

<style scoped>
.container {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: 500px 500px;
    grid-gap: 10px;
}

.content-section {
    grid-row: 1;
    grid-column: 1;
    /* section 内容纵向排列 */
    display:flex;
    flex-direction: column;
    align-items: center;
}
.style-section {
    grid-row: 1;
    grid-column: 2/4;

    display:flex;
    flex-direction: column;
    align-items: center;
}
.result-section {
    grid-row: 2;
    grid-column: 2;

    display:flex;
    flex-direction: column;
    align-items: center;
}
    .section-main {
        flex-grow: 1; /* 占满flex主轴剩余高度 */
        align-self: stretch; /* 横向拉伸 */
        /* section-main里的subsection 横向排列 */
        display: flex;
    }
        .section-main .subsection{
            flex-grow: 1; /* 占满剩余高度 */
            width: 100%;
            /* subsection里的 纵向排列 */
            display: flex;
            flex-direction: column;
        }


.center-container {
    flex-grow: 1; /* 占满剩余高度 */
    
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: #ffffff;
    padding: 0.5rem;

    display: flex;
    justify-content: center;
    align-items: center;
}

.center-container img {
    border: 1px solid #eee;
    border-radius: 0.2rem;
    width: 100%;
    height: 100%;
    object-fit: cover;
    overflow: hidden;
}

.center-container .image-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    grid-template-rows: 100px 100px 100px;
    grid-gap: 5px;
}

.center-container .image-grid .image-item {
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.055);
}

.center-container .image-grid .image-item img {
    width: 100%;
}

img.selected {
    box-sizing: border-box;
    border: 4px solid #0088cc;
}

/*
input[type="file"] {
  display: none;
}
*/
</style>
