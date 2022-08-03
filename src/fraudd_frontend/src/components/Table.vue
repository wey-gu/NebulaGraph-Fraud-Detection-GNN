<template>
    <table-lite :has-checkbox="true" :is-loading="table.isLoading" :is-re-search="table.isReSearch"
        :columns="table.columns" :rows="table.rows" :total="table.totalRecordCount" :messages="table.messages"
        @is-finished="isFinished" @return-checked-rows="checkedRows">
    </table-lite>
</template>

<script>
import { ref, onMounted, defineComponent, reactive, computed } from "vue";
import TableLite from "vue3-table-lite";
import { io } from 'socket.io-client'


export default defineComponent({
    name: "App",
    components: {
        TableLite,
    },
    setup() {

        const socket = io('http://localhost:9000');

        const rows = ref([]);
        let isLoading = ref(false);
        const table = reactive({
            isLoading: false,
            isReSearch: false,
            columns: [
                {
                    label: "ID",
                    field: "id",
                    width: "3%",
                    sortable: true,
                    isKey: true,
                },
                {
                    label: "Name",
                    field: "name",
                    width: "10%",
                    sortable: true,
                },
                {
                    label: "f0",
                    field: "f0",
                    width: "3%",
                    sortable: true,
                },
                {
                    label: "f1",
                    field: "f1",
                    width: "3%",
                    sortable: true,
                },
                {
                    label: "f2",
                    field: "f2",
                    width: "3%",
                    sortable: true,
                },
                {
                    label: "Is Fraud",
                    field: "is_fraud",
                    width: "10%",
                },
            ],
            rows: rows,
            totalRecordCount: 10,
            messages: {
                pagingInfo: "The {0}-{1} of {2} Reviews",
                pageSizeChangeLabel: "Row count: ",
                gotoPageLabel: " Go to page:",
                noDataAvailable: "No Valid Data",
            },
        });
        socket.on('new_review', (message) => {
            let data = JSON.parse(message);
            rows.value.push(
                {
                    id: data.vertex_id,
                    name: `review ${data.vertex_id}`,
                    f0: data.feat[0],
                    f1: data.feat[1],
                    f2: data.feat[2],
                    f3: data.feat[3],
                    f4: data.feat[4],
                    is_fraud: data.is_fraud,
                }
            );
            isLoading = true;
        })

        onMounted(() => {
            console.log("onMounted: starting...")
            socket.emit("My Event", "Hello from the client!")
        })

        const isFinished = () => {
            table.isLoading = isLoading
        };
        const checkedRows = (rowsKey) => {
            console.log(rowsKey);
        };
        return {
            table,
            isFinished,
            checkedRows,
            rows: computed(() => rows),
            isLoading: computed(() => isLoading)
        };
    },
});

</script>

<style>
#app {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
}
</style>