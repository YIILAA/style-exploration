<template>
  <img :src="this.src"
        :class=selectedStr
        alt=""
        @click="onSelect">
</template>

<script>
export default {
    props: ["id", "src", "selected"],

    data() {
        return {
            selectedStr: ''
        }
    },

    methods: {
        onSelect() {
            // 点击一下 改变选中状态
            this.selectedStr == '' ? this.selectedStr='selected' : this.selectedStr = '' ;
            this.$emit("clicked", this.id); // 触发父组件的clicked事件，传递参数为id
        },

        
        /*
        classThumbnail(selectedId, imageId) { // 当前图片被选中，则其class为`selected`，用于改变css样式
            if (selectedId === imageId) {
                return `selected`
            }
            return ``
        }
        */
    }
};
</script>

<style scoped>
</style>