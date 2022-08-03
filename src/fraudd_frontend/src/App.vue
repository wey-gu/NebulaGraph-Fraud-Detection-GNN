<template>
  <!-- Nav_bar -->
  <div class="container">
    <div class="navbar">
      <div class="navbar-inner">
        <div class="container">
          <img
            src="https://user-images.githubusercontent.com/1651790/182613254-41294e1a-0c81-463c-8369-74cefaa4bb7c.png"
            width="100%" />
        </div>
      </div>
    </div>
  </div>
  <Table />
</template>

<script>
import Table from "./components/Table.vue";

export default {
  name: 'App',
  components: {
    Table
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

* {
  transition: all 0.3s ease-in-out;
}

.container {
  clear: both;
  overflow: auto;
}

.logo img {
  float: left;
  border-radius: 3px;
  overflow: hidden;
}

.line {
  clear: both;
  width: 100%;
  height: 10px;
  background-color: rgb(187, 188, 192);
  padding: 1px 0;
  margin-bottom: 10px;
}

.title {
  font-size: 1.5em;
  font-weight: light;
  color: #cbcbdc;
  margin-bottom: 10px;
}
</style>
