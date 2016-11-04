%{
	Autor: Daniel González Alonso
	Fecha: 30/10/2016
%}

clear;

%%%% Constantes %%%%
global MAP_ROWS			= 12;
global MAP_COLS			= 8;
global NUM_EPOCAS		= 50;
global RUTA_FICHERO1	= 'distancias_entrenamiento.csv';
global RUTA_FICHERO2	= 'distancias_test.csv';


%%%% Funciones %%%%
% Funcion que escribe las distancias de las neuronas a las instancias de
% m_datos junto con su clase en un fichero CSV
function EscribeCSV(ruta_fichero, m_datos, m_pesos, m_salida_esperada)
	global MAP_ROWS;
	global MAP_COLS;

	if (~ismatrix(m_datos) || ~ismatrix(m_pesos) || ~ismatrix(m_salida_esperada))
		error('Data tiene que ser una matriz')
	end;

	% Cabecera
	cabecera = '';
	for (i = 1:MAP_ROWS * MAP_COLS)
		cabecera = [cabecera, 'neurona', num2str(i), ','];
	end;
	cabecera = [cabecera, 'clase\n'];

	% Creamos el archivo csv
	archivo = fopen(ruta_fichero, 'w');
	fprintf(archivo, cabecera);
	fclose(archivo);

	% Escribimos los datos
	for (dato = 1:size(m_datos,1))
		% calculamos las distancias de las neuronas a la entrada actual
		v_x = m_datos(dato,:);	% muestra actual
		v_y = v_x * m_pesos';
		
		% calculamos la de la entrada actual
		for (i = 1:10)
		  if (m_salida_esperada(dato,i) == 0.9)
				% utilizamos el filtro de elevar a la cuarta y concatenamos
				% las distancias con la clase actual
				m_output(dato,:) = [v_y.^4, i];
				break;
		  end;
		end;
	end;

	% Añadimos los datos al fichero csv
	dlmwrite(ruta_fichero, m_output, '-append');

end;


%%%% Obtenemos los datos %%%%
m_input = dlmread('digitos.entrena.normalizados.txt');
m_salida_esperada	= m_input(2:2:rows(m_input),1:1:10);
m_datos				= m_input(1:2:rows(m_input),:);

% Normalizacion extendida
m_datos			= [m_datos, ones(rows(m_datos),1)];
m_datos_norm	= m_datos./sqrt( sum((m_datos.^2),2) );

n_datos			= size(m_datos_norm,1);
n_propiedades   = size(m_datos_norm,2);


%%%% Aprendizaje %%%%
% Creamos la matriz que contendrá los pesos
for (i = 1:MAP_ROWS * MAP_COLS)
	m_pesos(i,:) = rand(1,n_propiedades) - 0.5*ones(1,n_propiedades);
	m_pesos(i,:) = m_pesos(i,:)./sqrt( sum((m_pesos(i,:).^2),2) );
end;

% Calculamos el radio inicial
radio = idivide(min(MAP_ROWS, MAP_COLS), 2);

% Aprendizaje
for (epoca = 1:NUM_EPOCAS)

	n_alpha = 25/(1 + epoca/n_datos);
	for (dato = 1:n_datos)

		% Calculamos la neurona ganadora
		v_x = m_datos_norm(dato,:);	% muestra actual
		v_y = v_x * m_pesos';
		[n_ganadora, index_ganadora] = max(v_y);
		[i_ganadora, j_ganadora] = ind2sub([MAP_ROWS, MAP_COLS], index_ganadora);

		% Calculamos los siguientes pesos de las neuronas pertencientes a N(I)
		for (i = i_ganadora - radio:i_ganadora + radio)
			% Calculamos la i de la neurona
			if (i < 1)
				i_cur_neurona = MAP_ROWS + i;
			elseif(i > MAP_ROWS)
				i_cur_neurona = mod(i-1, MAP_ROWS) + 1;
			else
				i_cur_neurona = i;
			end;

			for (j = j_ganadora - radio:j_ganadora + radio)
				% Calculamos la j de la neurona
				if (j < 1)
					j_cur_neurona = MAP_COLS + j;
				elseif(j > MAP_COLS)
					j_cur_neurona = mod(j-1, MAP_COLS) + 1;
				else
					j_cur_neurona = j;
				end;

				% Calculamos el indice de la neurona en la matriz de pesos
				index_cur_neurona = sub2ind([MAP_ROWS, MAP_COLS], i_cur_neurona, j_cur_neurona);

				% Calculamos el nuevo peso de la neurona
				v_sig_pesos = m_pesos(index_cur_neurona,:) + n_alpha*v_x;
				m_pesos(index_cur_neurona,:) = v_sig_pesos./sqrt( sum((v_sig_pesos.^2),2) );
			end;
		end;% fin pesos

	end;% fin epoca

	if (radio > 0)
		radio -= 1;
	end;

end;% fin epocas


%%%% Archivo CSV I %%%%
EscribeCSV(RUTA_FICHERO1, m_datos_norm, m_pesos, m_salida_esperada);


%{
%%%% ETIQUETADO POR NEURONAS %%%%
m_etiquetas = zeros(MAP_ROWS * MAP_COLS, 10);
for (index_cur_neurona = 1:MAP_ROWS * MAP_COLS)
	% Calculamos la muestra de la entrada con la menor distancia a la neurona actual
	v_y = m_datos_norm * m_pesos(index_cur_neurona,:)';
	[n_mejor, index_mejor] = max(v_y);

	% Etiquetamos la neurona ganadora con la categoria de la muestra actual
	m_etiquetas(index_cur_neurona,:) = m_salida_esperada(index_mejor,:);
end;
%}


% Fichero de test
m_input = dlmread('digitos.test.normalizados.txt');

m_salida_esperada	= m_input(2:2:rows(m_input),1:1:10);
m_datos				= m_input(1:2:rows(m_input),:);

% Normalizacion extendida
m_datos			= [m_datos, ones(rows(m_datos),1)];
m_datos_norm	= m_datos./sqrt( sum((m_datos.^2),2) );

n_propiedades   = size(m_datos_norm, 2);
n_datos			= size(m_datos_norm, 1);

%{
%%%% TEST %%%%
% Calculamos el numero de etiquetas que coinciden
n_aciertos = 0;
for (dato = 1:n_datos)
	% Calculamos la neurona ganadora
	v_x = m_datos_norm(dato,:);	% muestra actual
	v_y = v_x * m_pesos';
	[n_ganadora, index_ganadora] = max(v_y);

	% Comprobamos si las etiquetas coinciden
	if (m_salida_esperada(dato, :) == m_etiquetas(index_ganadora, :))
		n_aciertos++;
	end;
end;

porcentaje_aciertos = n_aciertos / n_datos
%}


%%%% Archivo CSV II %%%%
EscribeCSV(RUTA_FICHERO2, m_datos_norm, m_pesos, m_salida_esperada);
